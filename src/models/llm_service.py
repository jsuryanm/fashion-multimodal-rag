import logging 

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config.settings import settings

logger = logging.getLogger(__name__)


def build_groq_llm(groq_api_key: str = settings.groq_api_key,
                   model_id: str = settings.groq_model_id,
                   temperature: float = settings.groq_temperature,
                   max_tokens: int = settings.groq_max_tokens) -> ChatGroq:
    
    llm = ChatGroq(model=model_id,
                   api_key=groq_api_key,
                   temperature=temperature,
                   max_tokens=max_tokens)
    
    return llm 

def build_fashion_messages(image_data_uri: str,
                           matched_items: list[dict],
                           is_exact_match: bool) -> list:
    """
    Build the LangChain message list for the fashion analysis prompt.
 
    Args:
        image_data_uri:  Base64 data URI — "data:image/jpeg;base64,..."
        matched_items:   List of metadata dicts from FAISS retrieval.
                         Each dict has Item Name, Price, Link, all_items.
        is_exact_match:  True if the best similarity score >= threshold.
 
    Returns:
        [SystemMessage, HumanMessage] ready to pass to ChatGroq.
    
    """
    all_items: list[dict] = []
    if matched_items and "all_items" in matched_items[0]:
        all_items = matched_items[0]["all_items"]
    
    else:
        all_items = [
            {"Item Name": m["Item Name"], "Price": m["Price"], "Link": m["Link"]}
            for m in matched_items
        ]

    items_md = "\n".join(
        f"- {item['Item Name']} (${item['Price']}): {item['Link']}"
        for item in all_items[:10]  # cap at 10 to stay within token limits
    )

    section_header = "MATCHED CATALOGUE ITEMS" if is_exact_match else "SIMILAR CATALOGUE ITEMS"
    
    match_note = ("This image closely matches an item in our catalogue" 
                  if is_exact_match 
                  else "No exact match found - showing the most visually similar items.")

    system_message = SystemMessage(content=(
        "You are a professional fashion analyst writing for a luxury retail catalogue. "
        "Produce objective, clinical product descriptions suitable for a retail database. "
        "Always respond in valid Markdown using exactly these four sections:\n\n"
        "## Garments\n"
        "List each visible clothing item with colour, pattern, and material.\n\n"
        "## Accessories\n"
        "List visible accessories (bags, jewellery, shoes, belts, etc.).\n\n"
        "## Overall Style\n"
        "One paragraph categorising the style (e.g. business casual, streetwear, athleisure).\n\n"
        f"## {section_header}\n"
        "Reproduce the item list provided verbatim — do not modify prices or links.\n\n"
        "Do NOT add any content outside these four sections."))
    
    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"Please analyse the fashion image.\n\n"
                    f"Note: {match_note}\n\n"
                    f"{section_header}:\n{items_md}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_data_uri},
            },
        ]
    )

    return [system_message,human_message]
