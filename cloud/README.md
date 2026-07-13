# MnemonicAI Cloud — multi-tenant SaaS

Turns Aerith into a legitimate web-served AI product: sign in with an access
key, get a private isolated instance with its own memory + brain monitor,
manage billing, upload your codebase.

## Architecture (hybrid, per user decision)
- **Free / Starter:** share one always-on A40 pod; each user's memories,
  files, and monitor are isolated by `tenants.py` (separate DB per tenant,
  no cross-tenant code path).
- **Pro:** dedicated GPU pod, spun up on sign-in, stopped after 30 min idle.
- **Billing:** Stripe subscriptions (Starter $19 / Pro $79), hosted portal
  for cancel + payment-method updates.

## Status
| Piece | State |
|---|---|
| Tenant isolation (`tenants.py`) | ✅ built + tested (auth, isolation, no key-leak, quotas) |
| Web GUI (`webapp/index.html`) | ✅ built — 7 themes, sign-in, chat, per-user brain monitor, account/billing, support, upload-with-virus-warning; verified in preview |
| Stripe billing | ✅ checkout/portal/webhook built + verified against test account; products created |
| Upload + ClamAV virus scan | ✅ live on pod — clean stored, EICAR rejected (fails closed) |
| Dedicated Pro pod lifecycle | ✅ podlife.py — create-on-signin + 30min idle stop |

## Files
- `tenants.py` — tenant store: access-key→user, tier/quota, isolated paths.
- `webapp/index.html` — self-contained SPA (no build step). Themes in
  `[data-theme]` CSS. Currently mocks API calls (marked `DEMO`); wiring
  points to the tenant gateway are commented.

## Themes
aurora (default), rose, forest, solar, ocean, mono, daylight — chosen to
span a range of tastes; switchable live from the header, persisted per user.

## To go live (remaining wiring)
1. Stripe keys → env; add `/billing/*` webhook to flip `tenant.tier`.
2. Front the shared pod's MnemonicAI with a tenant gateway that reads the
   access key, loads that tenant's `memory.db`, filters monitor events.
3. ClamAV (`clamd`) on the pod; scan uploads before they land in the
   tenant's `uploads/`.
4. Pro pod controller: RunPod API create-on-signin, idle-timer stop.
