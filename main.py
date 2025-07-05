import os
import hashlib
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from bs4 import BeautifulSoup
import requests
import nest_asyncio
from google import genai
from playwright.async_api import async_playwright
from huggingface_hub import InferenceClient
import sys

# Fix for Windows event loop policy
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ---------------------- INIT ----------------------
load_dotenv()
nest_asyncio.apply()  # Ensures compatibility in Jupyter/interactive environments

# Initialize Gemini (Google GenAI) client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone vector index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "query-cache-vector"
index = pc.Index(INDEX_NAME)

# ---------------------- EMBEDDING ----------------------
# Generate embedding vector for a given query
def get_embedding(text: str) -> list:
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    [embedding] = result.embeddings
    return embedding.values

# Hash the query text for a unique ID in Pinecone
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

# ---------------------- CACHING ----------------------
# Store new query and its result in Pinecone
def store_query_result(query: str, result: str):
    embedding = get_embedding(query)
    query_id = hash_text(query)

    index.upsert([
        {
            "id": query_id,
            "values": embedding,
            "metadata": {
                "query": query,
                "result": result
            }
        }
    ])

# Check if a similar query already exists in Pinecone
def find_similar_query(query: str, threshold: float = 0.70):
    embedding = get_embedding(query)
    response = index.query(vector=embedding, top_k=1, include_metadata=True)

    if response.matches and response.matches[0].score >= threshold:
        return response.matches[0].metadata["result"]
    
    return None

# ------------------- WEB SEARCH -------------------
# Use DuckDuckGo and Playwright to get top search results and scrape pages
async def search_web_DuckDuckgo(query, top_k=5):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        page = await context.new_page()

        # Perform search
        await page.goto("https://duckduckgo.com/")
        await page.wait_for_selector("input[name='q']", timeout=20000)
        await page.fill("input[name='q']", query)
        await page.keyboard.press("Enter")
        await page.wait_for_selector("a[data-testid='result-title-a']", timeout=10000)

        results = []
        anchors = await page.query_selector_all("a[data-testid='result-title-a']")

        for anchor in anchors:
            href = await anchor.get_attribute("href")
            title = await anchor.inner_text()

            if not href or "youtube.com" in href:
                continue

            # Open and scrape each result page
            detail_page = await context.new_page()
            try:
                await detail_page.goto(href, timeout=20000)
                content = await detail_page.content()
                soup = BeautifulSoup(content, "html.parser")

                heading = soup.find("h1")
                subheadings = [h.get_text(strip=True) for h in soup.find_all(["h2", "h3"])]
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 50]

                results.append({
                    "title": title.strip(),
                    "url": href,
                    "heading": heading.get_text(strip=True) if heading else "",
                    "subheadings": subheadings,
                    "paragraphs": paragraphs
                })
            except Exception as e:
                print(f"Failed to scrape {href}: {e}")
            finally:
                await detail_page.close()

            if len(results) >= top_k:
                break

        await browser.close()
        return results

# ---------------------- TEXT SCRAPING ----------------------
# Fallback method to extract text using requests
def scrape_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        return "\n".join(p.text for p in soup.find_all('p'))[:5000]
    except Exception as e:
        return f"Failed to scrape: {e}"

# ---------------------- SUMMARIZATION ----------------------
# Summarize combined web content using HuggingFace LLM
def summarize_combined_text(text, system_prompt):
    client = InferenceClient(provider="together", api_key=os.getenv("HUGGINGFACE_API_KEY"))

    prompt = f"""
    Use the following system prompt to generate a detailed and unified summary:

    SYSTEM PROMPT: {system_prompt}

    COMBINED CONTENT:
    {text[:8000]}
    """

    stream = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=True,
    )

    output = ""
    for chunk in stream:
        output += chunk.choices[0].delta.content or ""
    return output.strip()

# ---------------------- QUERY HANDLER ----------------------
# Main function to handle end-to-end query workflow
async def handle_query(query: str):
    print(f"\nUser Query: {query}")

    # Check for cached result
    cached = find_similar_query(query)
    if cached:
        print("✅ Found cached result.")
        return cached

    print("🔍 Searching web...")

    # Perform live search
    search_results = await search_web_DuckDuckgo(query)

    print(f"\n🔍 Top results for: '{query}'\n")
    combined_text = ""
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result['title']} - {result['url']}")
        combined_text += scrape_text(result['url']) + "\n\n"

    print("🧠 Generating summary from combined content...")
    system_prompt = (
        "You are an intelligent and reliable web agent. "
        "The user has asked: \"{user_query}\". "
        "You have access to multiple top-ranked web pages related to this query. "
        "Your job is to read all the content, understand the topic deeply, and generate a single, accurate, detailed, and well-structured summary. "
        "Ensure the information is factual, relevant to the user's intent, and avoids speculation or unrelated details. "
        "Focus on the main subject, highlight important points, and provide a clear explanation based only on the retrieved content."
    )
    
    # Summarize all collected content
    summary = summarize_combined_text(combined_text, system_prompt)

    if not summary:
        return "❌ No URLs found."

    # Cache the result for future similar queries
    print("💾 Caching result in Pinecone...")
    store_query_result(query, summary)

    return summary

# ---------------------- MAIN ----------------------
# CLI Entry Point
async def main():
    user_query = input("Enter your search query: ")
    print("📄 FINAL SUMMARY:")
    print(await handle_query(user_query))


if __name__ == "__main__":
    asyncio.run(main())
