"""
Stripe webhook handler for MnemonicAi.
When a customer pays → create their account + API key.
When they cancel → deactivate their key.
"""

import sqlite3
import json
from datetime import datetime, timezone
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from app.auth import generate_api_key, init_auth_db

DB_PATH = "mnemonicai_data/auth.db"

# Tier configuration
TIER_CONFIG = {
    "price_starter": {  # Replace with your actual Stripe Price ID
        "tier": "starter",
        "monthly_limit": 5000,
        "name": "Starter",
    },
    "price_pro": {  # Replace with your actual Stripe Price ID
        "tier": "pro",
        "monthly_limit": 999999,  # Effectively unlimited
        "name": "Pro",
    },
    # Add your actual price_xxx IDs here
}


async def handle_stripe_webhook(request: Request):
    """
    Webhook endpoint: POST /stripe-webhook
    Configure this URL in your Stripe Dashboard → Webhooks.
    """
    import stripe

    # You'll set this as an environment variable
    stripe.api_key = "sk_live_YOUR_KEY"  # Use env var in production!
    webhook_secret = "whsec_YOUR_WEBHOOK_SECRET"  # Use env var!

    # Read the raw body (Stripe needs it for signature verification)
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid payload"})
    except stripe.error.SignatureVerificationError:
        return JSONResponse(status_code=400, content={"error": "Invalid signature"})

    # ─── Handle checkout.session.completed ───────────────────
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        email = session.get("customer_details", {}).get("email")
        stripe_customer_id = session.get("customer")
        
        # Find the tier from the price ID
        line_items = stripe.checkout.Session.list_line_items(session["id"])
        price_id = line_items["data"][0]["price"]["id"] if line_items["data"] else None
        tier_config = TIER_CONFIG.get(price_id, {"tier": "starter", "monthly_limit": 5000})

        # Create or update user
        user_id = f"user_{stripe_customer_id}"
        conn = sqlite3.connect(DB_PATH)
        
        # Insert user (or update if exists)
        conn.execute(
            """
            INSERT INTO users (id, email, stripe_customer_id, created_at, tier, status)
            VALUES (?, ?, ?, ?, ?, 'active')
            ON CONFLICT(id) DO UPDATE SET tier=excluded.tier, status='active'
            """,
            (user_id, email, stripe_customer_id,
             datetime.now(timezone.utc).isoformat(), tier_config["tier"])
        )

        # Generate API key
        full_key, key_hash, key_prefix = generate_api_key()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        conn.execute(
            """
            INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, 
                                  created_at, status, monthly_limit, 
                                  requests_this_month, reset_date)
            VALUES (?, ?, ?, ?, 'primary', ?, 'active', ?, 0, ?)
            """,
            (f"key_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
             user_id, key_hash, key_prefix,
             datetime.now(timezone.utc).isoformat(),
             tier_config["monthly_limit"], today)
        )
        conn.commit()
        conn.close()

        # Email the API key to the customer
        # (You can use Resend, SendGrid, or your existing email setup)
        send_welcome_email(email, full_key, tier_config["name"])

        print(f"✅ New customer: {email} | Tier: {tier_config['tier']} | Key: {key_prefix}")

    # ─── Handle subscription deleted (cancellation) ──────────
    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        stripe_customer_id = subscription.get("customer")
        user_id = f"user_{stripe_customer_id}"

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE users SET status = 'cancelled' WHERE id = ?",
            (user_id,)
        )
        conn.execute(
            "UPDATE api_keys SET status = 'inactive' WHERE user_id = ?",
            (user_id,)
        )
        conn.commit()
        conn.close()

        print(f"❌ Cancelled: {user_id}")

    # ─── Handle payment failed ───────────────────────────────
    elif event["type"] == "invoice.payment_failed":
        invoice = event["data"]["object"]
        stripe_customer_id = invoice.get("customer")
        user_id = f"user_{stripe_customer_id}"

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE api_keys SET status = 'suspended' WHERE user_id = ? AND status = 'active'",
            (user_id,)
        )
        conn.commit()
        conn.close()

        print(f"⚠️ Payment failed: {user_id}")

    return JSONResponse(status_code=200, content={"received": True})


def send_welcome_email(email: str, api_key: str, tier_name: str):
    """
    Send the customer their API key after payment.
    Use Resend (free tier), SendGrid, or your existing email setup.
    """
    # Example using Resend (free up to 100 emails/day)
    try:
        import resend
        resend.api_key = "re_YOUR_KEY"  # Use env var!

        resend.Emails.send({
            "from": "MnemonicAi <welcome@mnemonicai.org>",
            "to": [email],
            "subject": f"Your MnemonicAi API Key ({tier_name} plan)",
            "html": f"""
                <h2>Welcome to MnemonicAi! 🧠</h2>
                <p>Your {tier_name} plan is now active.</p>
                <p>Here's your API key (keep it secret):</p>
                <pre style="background: #f4f4f4; padding: 16px; border-radius: 8px; font-size: 14px;">{api_key}</pre>
                <h3>Quick Start</h3>
                <p>Just point your OpenAI client to MnemonicAi:</p>
                <pre style="background: #f4f4f4; padding: 16px; border-radius: 8px;">
from openai import OpenAI

client = OpenAI(
    api_key="{api_key}",
    base_url="https://api.mnemonicai.org/v1"
)

response = client.chat.completions.create(
    model="ornith-1.0-9b",
    messages=[{{"role": "user", "content": "My name is Alice"}}]
)

# Next session — it remembers:
response = client.chat.completions.create(
    model="ornith-1.0-9b",
    messages=[{{"role": "user", "content": "What's my name?"}}]
)
# → "Your name is Alice."
                </pre>
                <p>Docs: https://mnemonicai.org/docs</p>
                <p>Dashboard: https://mnemonicai.org/dashboard</p>
            """,
        })
    except Exception as e:
        print(f"Email send failed: {e}")
        # Fallback: log it so you can manually send
        with open("mnemonicai_data/logs/api_keys_to_send.txt", "a") as f:
            f.write(f"{email}: {api_key}\n")
