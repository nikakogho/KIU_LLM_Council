## Council Checking Script
Example CLI run

`python scripts/council_preview.py "Design a robust cache invalidation plan for a CDN-backed API."`

Result:

Default judge: openai | gpt-5-nano
Solvers: xai, anthropic, gemini

[Phase 1] Drafting...

--- Draft ready: gemini ---
## Robust Cache Invalidation Plan for CDN-Backed API

This plan outlines a multi-layered approach to cache invalidation for a CDN-backed API, prioritizing data freshness, performance, and operational simplicity.

### Core Principles:

1.  **Event-Driven Invalidation:** Trigger invalidation based on actual data changes, not on fixed TTLs alone.
2.  ...

--- Draft ready: anthropic ---
# Robust Cache Invalidation Plan for CDN-Backed API

## Executive Summary
A multi-layered invalidation strategy combining time-based expiration, event-driven purging, and smart cache headers to balance performance with data freshness.

---

## Architecture Overview

'''
Client → CDN Edge → Origin API
         ↓
    Cache Layer (TTL + Headers)
     ...

--- Draft ready: xai ---
Below is a robust cache invalidation plan for a CDN-backed API. This plan is designed to balance performance, reliability, and cost, ensuring that cached content is updated promptly when the underlying data changes. I'll break it down into key sections: the proposed solution with steps, tradeoffs, and potential pitfalls. This approach draws from be...

[Phase 2] Peer reviews...

--- Review ready: gemini -> anthropic --- overall=8

--- Review ready: gemini -> xai --- (invalid JSON: No JSON object found.)

--- Review ready: anthropic -> xai --- overall=8

--- Review ready: anthropic -> gemini --- overall=7

--- Review ready: xai -> anthropic --- overall=8

--- Review ready: xai -> gemini --- overall=8

[Phase 3] Revisions...

--- Revision ready: gemini ---
## Robust Cache Invalidation Plan for CDN-Backed API

This plan outlines a multi-layered approach to cache invalidation for a CDN-backed API, prioritizing data freshness, performance, and operational simplicity. It addresses invalidation at the application, CDN edge, and client levels, incorporating robust error handling, monitoring, and security.
...

--- Revision ready: anthropic ---
# Robust Cache Invalidation Plan for CDN-Backed API

## Executive Summary
A multi-layered invalidation strategy combining time-based expiration, event-driven purging, and smart cache headers to balance performance with data freshness.     

---

## Architecture Overview

'''
Client → CDN Edge → Origin API
         ↓
    Cache Layer (TTL + Headers)
     ...

--- Revision ready: xai ---
Below is the revised robust cache invalidation plan for a CDN-backed API. This updated plan incorporates enhancements based on feedback, ensuring a more comprehensive approach that addresses monitoring completeness, cache coherency, resilience strategies, cascading invalidations, cost implications, security, and edge cases. The plan continues to ba...

[Phase 4] Judge decision...

==============================
WINNER: anthropic
==============================

# Robust Cache Invalidation Plan for CDN-Backed API

## Executive Summary
A multi-layered invalidation strategy combining time-based expiration, event-driven purging, and smart cache headers to balance performance with data freshness.     

---

## Architecture Overview

'''
Client → CDN Edge → Origin API
         ↓
    Cache Layer (TTL + Headers)
         ↓
    Event Bus (Invalidation Triggers)
         ↓
    Cache Invalidation Service
         ↓
    Origin Database
'''

---

## Core Strategy: Three-Tier Invalidation

### **Tier 1: HTTP Cache Headers (Passive)**
**Implementation:**
'''
Cache-Control: public, max-age=300, s-maxage=3600
ETag: "abc123def456"
Last-Modified: Wed, 21 Oct 2024 07:28:00 GMT
Vary: Accept-Encoding, Authorization
'''

**Rules by Content Type:**
- **Static assets** (images, CSS): `max-age=31536000` (1 year) + versioned URLs
- **API responses** (user data): `max-age=60` (1 minute)
- **Real-time data** (prices, inventory): `max-age=5, s-maxage=30`
- **Personalized content**: `Cache-Control: private` (CDN bypass)

**Tradeoff:** Simple but coarse-grained; stale data until TTL expires.

---

### **Tier 2: Event-Driven Purging (Active)**
**Trigger Points:**
1. **Data mutations** (POST/PUT/DELETE)
   - Publish to event bus immediately after write
   - Include affected resource IDs and related cache keys

2. **Scheduled invalidation**
   - Batch purge at off-peak hours
   - Useful for derived/aggregated data

3. **Manual purging**
   - Admin dashboard for emergency cache clears

**Implementation Pattern:**
'''
POST /api/users/123 (update user)
  ↓
Database transaction commits
  ↓
Emit: { event: "user.updated", id: 123, timestamp: "...", cacheKeys: ["users/123", "users/123/profile"] }
  ↓
Cache Invalidation Service receives event (async, non-blocking)
  ↓
CDN purge request: DELETE /cdn/cache/users/123
  ↓
Confirmation logged + metrics recorded
'''

**Latency Optimization:**
- Use asynchronous event processing to avoid blocking mutations (fire-and-forget pattern)
- Batch invalidation requests at CDN level (e.g., purge 50 keys in single API call)
- Employ high-performance message queues (Redis Streams, Kafka) with sub-50ms processing
- For ultra-sensitive data (prices, inventory), use Tier 1 with very short TTLs (5-10s) instead of relying solely on events

**Tradeoff:** Requires event infrastructure; adds 50-200ms latency to mutations (mitigated by async processing).

---

### **Tier 3: Conditional Requests (Validation)**
**Mechanism:**
- Client sends `If-None-Match: "etag-value"` or `If-Modified-Since`
- CDN/origin responds with `304 Not Modifie

--- Judge rationale ---
Anthropic's plan is the most coherent and actionable among the four, offering a clear architecture and practical invalidation mechanisms, with thoughtful tradeoffs, making it the best overall despite minor truncation.
