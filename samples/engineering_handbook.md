# Engineering Handbook

## Code review
Every pull request needs at least two approvals before merging. Authors should keep pull requests under 400 lines of changed code. Reviewers must respond within one business day.

## On-call rotation
Engineers join the on-call rotation after six months. Each shift lasts one week and starts on Monday at 10:00. The primary on-call engineer must acknowledge a page within 15 minutes. Incidents rated SEV1 require a written postmortem within five business days.

## Deployments
Production deployments happen through the CI pipeline only. Deployments are frozen on Fridays after 14:00 and during the last week of December. Every release must have a rollback plan documented in the release ticket.

## Security
Secrets must never be committed to the repository. Use the secrets manager and rotate API keys every 90 days. Laptops must use full-disk encryption.
