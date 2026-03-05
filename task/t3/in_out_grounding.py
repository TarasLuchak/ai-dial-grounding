import asyncio
import json
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format.
# Uses adaptive vector-based input grounding + output API-based grounding.

SYSTEM_PROMPT = """You are a hobby extraction system. You analyze user profiles retrieved via RAG context
and extract hobbies that match the user's search query.

## Instructions:
1. Analyze the RAG CONTEXT containing user profiles (each with id and about_me)
2. Based on the USER QUESTION, identify which users have matching hobbies
3. Group the matching user IDs by the specific hobby they match
4. Only include hobbies that are relevant to the user's question
5. A user can appear under multiple hobbies if applicable

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}"""


class HobbyMatch(BaseModel):
    hobby: str = Field(description="The name of the hobby (e.g., 'hiking', 'rock climbing')")
    user_ids: list[int] = Field(description="List of user IDs that have this hobby")


class HobbySearchResult(BaseModel):
    matches: list[HobbyMatch] = Field(default_factory=list, description="List of hobby matches with user IDs")


user_client = UserClient()

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    azure_deployment="text-embedding-3-small-1",
    dimensions=384
)

llm_client = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    azure_deployment="gpt-4o",
    api_version=""
)


def format_user_document(user: dict[str, Any]) -> str:
    """Embed only id and about_me to reduce context window."""
    return f"User:\n  id: {user.get('id')}\n  about_me: {user.get('about_me', '')}"


async def create_vectorstore(users: list[dict[str, Any]]) -> Chroma:
    """Cold start: create Chroma vectorstore from all users."""
    documents = []
    for user in users:
        doc = Document(
            page_content=format_user_document(user),
            id=str(user['id'])
        )
        documents.append(doc)

    # Batch embed documents (100 per batch)
    vectorstore = Chroma(embedding_function=embeddings)
    batches = [documents[i:i + 100] for i in range(0, len(documents), 100)]
    for batch in batches:
        await vectorstore.aadd_documents(batch)

    return vectorstore


async def sync_vectorstore(vectorstore: Chroma) -> None:
    """Adaptive sync: compare current users with vectorstore, add new and remove deleted."""
    # Get current users from User Service
    current_users = user_client.get_all_users()
    current_ids = {str(user['id']) for user in current_users}

    # Get all IDs currently in Chroma
    stored = vectorstore.get()
    stored_ids = set(stored['ids']) if stored['ids'] else set()

    # Find new and deleted users
    new_ids = current_ids - stored_ids
    deleted_ids = stored_ids - current_ids

    # Delete removed users
    if deleted_ids:
        vectorstore.delete(ids=list(deleted_ids))
        print(f"Removed {len(deleted_ids)} deleted users from vectorstore")

    # Add new users
    if new_ids:
        new_users = [u for u in current_users if str(u['id']) in new_ids]
        new_docs = [
            Document(page_content=format_user_document(u), id=str(u['id']))
            for u in new_users
        ]
        await vectorstore.aadd_documents(new_docs)
        print(f"Added {len(new_ids)} new users to vectorstore")

    if not new_ids and not deleted_ids:
        print("Vectorstore is up to date")


def retrieve_context(vectorstore: Chroma, query: str, k: int = 20) -> str:
    """Similarity search on Chroma to find relevant user profiles."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    context_parts = []
    for doc, score in results:
        context_parts.append(doc.page_content)
        print(f"Score: {score:.4f} | {doc.page_content[:80]}...")
    return "\n\n".join(context_parts)


def extract_hobbies(context: str, query: str) -> HobbySearchResult:
    """Use LLM with structured output to extract hobbies and user IDs."""
    parser = PydanticOutputParser(pydantic_object=HobbySearchResult)
    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT.format(context=context, query=query))
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages).partial(
        format_instructions=parser.get_format_instructions()
    )
    result: HobbySearchResult = (prompt | llm_client | parser).invoke({})
    return result


async def ground_output(hobby_result: HobbySearchResult) -> dict[str, list[dict[str, Any]]]:
    """Output grounding: fetch full user info for each ID to verify and enrich."""
    grounded = {}
    for match in hobby_result.matches:
        users = []
        for user_id in match.user_ids:
            try:
                user_data = await user_client.get_user(user_id)
                users.append(user_data)
            except Exception as e:
                print(f"User ID {user_id} not found (hallucination filtered): {e}")
        if users:
            grounded[match.hobby] = users
    return grounded


async def main():
    print("Loading all users and creating vectorstore (cold start)...")
    all_users = user_client.get_all_users()
    vectorstore = await create_vectorstore(all_users)
    print(f"Vectorstore ready with {len(all_users)} users.\n")

    print("HOBBIES SEARCHING WIZARD")
    print("Query samples:")
    print(" - I need people who love to go to mountains")
    print(" - Find users interested in painting and art")
    print(" - Who likes cooking?")

    while True:
        user_question = input("> ").strip()
        if user_question.lower() in ['quit', 'exit']:
            break

        # Step 1: Adaptive sync - update vectorstore with new/deleted users
        print("\n--- Syncing vectorstore ---")
        await sync_vectorstore(vectorstore)

        # Step 2: Retrieve context via similarity search
        print("\n--- Retrieving context ---")
        context = retrieve_context(vectorstore, user_question)

        if not context:
            print("No relevant users found.")
            continue

        # Step 3: Extract hobbies and user IDs via LLM (NER)
        print("\n--- Extracting hobbies ---")
        hobby_result = extract_hobbies(context, user_question)
        print(f"LLM extracted: {hobby_result}")

        if not hobby_result.matches:
            print("No matching hobbies found.")
            continue

        # Step 4: Output grounding - fetch full user info
        print("\n--- Output grounding (fetching full user data) ---")
        grounded = await ground_output(hobby_result)

        # Step 5: Present results
        print(f"\n--- Results ---")
        print(json.dumps(grounded, indent=2, default=str))
        print()


if __name__ == "__main__":
    asyncio.run(main())
