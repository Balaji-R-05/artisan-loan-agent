from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
import random, json
import dotenv

dotenv.load_dotenv()

app = FastAPI()

# ---------- TOOLS (robust to string inputs) ----------

def sentiment_analysis(reviews) -> dict:
    """
    Analyze reviews for sentiment.
    Handles input as list or JSON string.
    """
    if isinstance(reviews, str):
        try:
            reviews = json.loads(reviews)
        except json.JSONDecodeError:
            # If it's not valid JSON, treat as single review
            reviews = [reviews]
    elif not isinstance(reviews, list):
        reviews = [str(reviews)]

    scores = []
    for r in reviews:
        r_str = str(r).lower()
        if "poor" in r_str or "late" in r_str or "bad" in r_str:
            scores.append(-0.5)
        elif "excellent" in r_str or "great" in r_str or "amazing" in r_str:
            scores.append(0.9)
        else:
            scores.append(0.8)
    
    avg = sum(scores) / len(scores) if scores else 0
    return {"average_sentiment": avg, "details": scores}


def market_trend_simulation(craft_type) -> dict:
    """
    Simulate market demand & seasonal index.
    """
    craft = str(craft_type)
    
    # Simple simulation based on craft type
    demand_multipliers = {
        "jewelry": 0.8,
        "pottery": 0.6,
        "textiles": 0.7,
        "woodwork": 0.5,
        "metalwork": 0.4
    }
    
    base_demand = demand_multipliers.get(craft.lower(), 0.6)
    demand_index = round(base_demand + random.uniform(-0.2, 0.3), 2)
    seasonal_index = round(random.uniform(0.3, 0.8), 2)
    
    return {
        "craft_type": craft,
        "demand_index": max(0.1, min(1.0, demand_index)),
        "seasonal_index": seasonal_index
    }


def collaboration_impact(count) -> dict:
    """
    Compute collaboration impact.
    Handles int, string, or dict input.
    """
    if isinstance(count, dict):
        # If agent passes a dict, extract the count value
        count = count.get('collaborations', count.get('count', 0))
    elif isinstance(count, str):
        try:
            count = int(count)
        except ValueError:
            count = 0
    elif not isinstance(count, int):
        count = 0
        
    score = min(1.0, 0.2 * count)
    return {"collaboration_score": score, "collaboration_count": count}


def transaction_consistency(data) -> dict:
    """
    Calculate transaction consistency.
    Handles dict, JSON string, or direct parameters.
    """
    # Handle different input formats
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {"transaction_score": 0, "error": "Invalid JSON format"}
    elif not isinstance(data, dict):
        return {"transaction_score": 0, "error": "Invalid input format"}

    total = data.get("total_orders", 0)
    completed = data.get("completed_orders", 0)
    
    # Ensure we have valid numbers
    try:
        total = int(total)
        completed = int(completed)
    except (ValueError, TypeError):
        return {"transaction_score": 0, "error": "Invalid order numbers"}

    completion_rate = completed / total if total > 0 else 0
    
    return {
        "transaction_score": round(completion_rate, 3),
        "total_orders": total,
        "completed_orders": completed,
        "completion_rate_percentage": round(completion_rate * 100, 1)
    }


# ---------- Wrap tools ----------
tools = [
    Tool(
        name="sentiment_analysis",
        func=sentiment_analysis,
        description="Analyze seller reviews and return sentiment scores. Pass reviews as a list or JSON string."
    ),
    Tool(
        name="market_trend_simulation",
        func=market_trend_simulation,
        description="Simulate demand and seasonal trends for craft type. Pass craft type as string."
    ),
    Tool(
        name="collaboration_impact",
        func=collaboration_impact,
        description="Compute score based on number of collaborations. Pass collaboration count as integer."
    ),
    Tool(
        name="transaction_consistency",
        func=transaction_consistency,
        description="Calculate transaction consistency score. Pass transaction data as dict with total_orders and completed_orders."
    )
]

# ---------- Gemini LLM ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# ---------- Request Model ----------
class LoanRequest(BaseModel):
    seller_id: str
    craft_type: str
    transaction_history: dict
    reviews: list
    collaborations: int


# ---------- Endpoint ----------
@app.post("/loan/evaluate")
async def evaluate_loan(req: LoanRequest):
    prompt = f"""
    You are an AI loan officer for an artisan marketplace.
    Given the data below, decide if the seller is eligible for a loan.

    Seller ID: {req.seller_id}
    Craft: {req.craft_type}
    Transaction history: {json.dumps(req.transaction_history)}
    Reviews: {json.dumps(req.reviews)}
    Collaborations: {req.collaborations}

    Use the available tools to gather insights:
    1. Use transaction_consistency with the transaction history data
    2. Use sentiment_analysis with the reviews data
    3. Use market_trend_simulation with the craft type
    4. Use collaboration_impact with the collaborations count

    After gathering all insights, compute a weighted score:
      40% transaction consistency
      30% sentiment
      20% market demand
      10% collaborations

    Risk tiers:
    - Low: score >= 0.7
    - Medium: 0.4 <= score < 0.7
    - High: score < 0.4

    Then return STRICT JSON only (no other text):
    {{
      "seller_id": "{req.seller_id}",
      "loan_eligibility": true/false,
      "risk_score": number,
      "risk_tier": "Low/Medium/High",
      "recommended_loan_amount": number,
      "batch_size": number,
      "reasoning": [list of strings explaining the decision]
    }}
    """

    try:
        result = await agent.arun(prompt)
        
        # Extract JSON from the result more aggressively
        result = str(result).strip()
        
        # Handle various markdown formats
        if "```json" in result:
            start = result.find("```json") + 7
            end = result.find("```", start)
            if end != -1:
                result = result[start:end].strip()
        elif "````json" in result:
            start = result.find("````json") + 8
            end = result.find("````", start)
            if end != -1:
                result = result[start:end].strip()
        elif result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        
        # Find JSON object if it's embedded in text
        if not result.startswith("{"):
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end > start:
                result = result[start:end]
        
        # Parse JSON response
        try:
            decision = json.loads(result)
            
            # Validate required fields
            required_fields = ["seller_id", "loan_eligibility", "risk_score", "risk_tier", 
                             "recommended_loan_amount", "batch_size", "reasoning"]
            
            for field in required_fields:
                if field not in decision:
                    decision[field] = get_default_value(field, req.seller_id)
                    
            return decision
            
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed: {json_err}")
            print(f"Raw result: {result}")
            
            # Try to extract values manually if JSON parsing fails
            decision = extract_values_manually(result, req.seller_id)
            decision["parsing_error"] = str(json_err)
            decision["raw_output"] = result
            return decision
        
    except Exception as e:
        print(f"General error: {str(e)}")
        # Return error response
        return {
            "seller_id": req.seller_id,
            "loan_eligibility": False,
            "risk_score": 0.0,
            "risk_tier": "High",
            "recommended_loan_amount": 0,
            "batch_size": 0,
            "reasoning": [f"Error occurred: {str(e)}"],
            "error": str(e)
        }


def get_default_value(field, seller_id):
    """Get default values for missing fields"""
    defaults = {
        "seller_id": seller_id,
        "loan_eligibility": False,
        "risk_score": 0.5,
        "risk_tier": "Medium",
        "recommended_loan_amount": 0,
        "batch_size": 0,
        "reasoning": ["Missing data - using defaults"]
    }
    return defaults.get(field, None)


def extract_values_manually(text, seller_id):
    """Manually extract values from text when JSON parsing fails"""
    import re
    
    decision = {
        "seller_id": seller_id,
        "loan_eligibility": False,
        "risk_score": 0.5,
        "risk_tier": "Medium",
        "recommended_loan_amount": 0,
        "batch_size": 0,
        "reasoning": ["Manual extraction due to JSON parsing failure"]
    }
    
    try:
        # Extract loan eligibility
        if "loan_eligibility" in text.lower():
            if "true" in text.lower():
                decision["loan_eligibility"] = True
        
        # Extract risk score
        risk_match = re.search(r'"risk_score":\s*([0-9.]+)', text)
        if risk_match:
            decision["risk_score"] = float(risk_match.group(1))
        
        # Extract risk tier
        tier_match = re.search(r'"risk_tier":\s*"([^"]+)"', text)
        if tier_match:
            decision["risk_tier"] = tier_match.group(1)
        
        # Extract recommended loan amount
        loan_match = re.search(r'"recommended_loan_amount":\s*([0-9]+)', text)
        if loan_match:
            decision["recommended_loan_amount"] = int(loan_match.group(1))
        
        # Extract batch size
        batch_match = re.search(r'"batch_size":\s*([0-9]+)', text)
        if batch_match:
            decision["batch_size"] = int(batch_match.group(1))
            
    except Exception as e:
        print(f"Manual extraction error: {e}")
    
    return decision


# ---------- Health Check ----------
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Artisan Loan Evaluation API"}


@app.get("/health")
async def health():
    return {"status": "ok"}