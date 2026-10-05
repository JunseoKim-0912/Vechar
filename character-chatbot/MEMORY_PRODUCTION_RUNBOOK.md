# Production memory rollout and recovery

Memory remains opt-in. Deploying this code does **not** change `MEMORY_PROVIDER=noop`.
Do not enable MemMachine until the migration, provider infrastructure, Queue
subscribers, and deletion behavior have been checked against a disposable
environment. Never run this migration against production from a test command.

1. Apply Alembic `0003_memory_ingestion` to a disposable PostgreSQL branch,
   verify one head, schema/metadata parity, upgrade/downgrade and row counts.
   Review the production database and perform its separately approved migration
   before deploying code that writes the new tables. Do not run migrations at
   application startup.
2. Provision the MemMachine project explicitly with
   `setup_memmachine_project.py --apply`, then run its read-only validation.
   The pinned client/common/server version is `0.3.9`; long-term episodic must
   be on and short-term episodic off. Setup is not done by chat or deployment.
3. Configure backend `MEMMACHINE_BASE_URL`, `MEMMACHINE_ORG_ID`,
   `MEMMACHINE_PROJECT_ID`, optional `MEMMACHINE_API_KEY`, and a private
   `CRON_SECRET` (at least 16 random characters). Keep them only in deployment
   secrets, not Git. Confirm `memory-ingest` and `memory-delete` subscribers
   are registered. `vercel.json` schedules a once-daily authenticated outbox
   reconciliation, compatible with Vercel Hobby; the normal Queue path runs
   immediately. A shorter reconciliation interval requires a plan that allows
   it. Queue/cron availability must be smoke-tested before activation.
4. Only in a later manual activation step, change `MEMORY_PROVIDER` from
   `noop` to `memmachine`. Confirm scoped cross-session retrieval and isolated
   user/character behavior with non-sensitive test records. Test character
   deletion while the provider is unavailable, then observe the durable
   deletion intent reach `completed` after recovery.

The DB is the authoritative outbox. Successful chat commits both its assistant
message and `memory_ingestions` row, then publishes an ID-only Queue message.
A failed publish does not undo chat; the reconciler republishes `queued` and
expired-lease work. Character deletion commits the `memory_deletions` tombstone
and DB deletion together. Deletion waits for active ingestion leases before
purging external episodes. No conversation-delete endpoint exists currently.

Operational checks (read-only SQL, with credentials supplied securely):

```sql
SELECT status, count(*) FROM memory_ingestions GROUP BY status;
SELECT status, count(*) FROM memory_deletions GROUP BY status;
SELECT id, provider, attempt_count, last_error_code, updated_at
FROM memory_deletions WHERE status = 'failed' ORDER BY updated_at DESC;
```

Alert on `failed` deletion tombstones, repeated `queued` rows, expired leases,
Queue publish errors, and Cron authorization errors. Do not log conversation
content or credentials. A failed tombstone is deliberately retained for manual
investigation; do not silently remove it. If an activated provider is later
disabled with `MEMORY_PROVIDER=noop`, previously recorded ingestion scopes
still create deletion tombstones. Pending operations pause until the matching
provider is explicitly re-enabled; inspect them before changing providers.

MemMachine 0.3.9 does **not** accept caller-chosen episode IDs or a single atomic
scope-delete operation. The adapter checks source-message metadata before each
episodic add, and lists/deletes scoped episode IDs in batches. A crash after a
provider write but before its result becomes visible to a retry can still make
a duplicate; a provider-side exactly-once guarantee is **not** claimed. A
provider index lag or non-atomic delete must be checked in a real integration
test before sensitive production data is enabled. Prompt hierarchy treats
retrieved memory as lower-priority reference context, not canonical truth;
generative temporal compliance also needs evaluation rather than assertion.
