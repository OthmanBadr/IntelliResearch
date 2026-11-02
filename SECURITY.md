# Credential incident and local deployment

Earlier public commits contained Google and Cohere credentials. Assume those
credentials have been copied. The refactor removes literal keys from the current
application, but **rotation/revocation must be completed in the provider accounts**.

1. Revoke or rotate every exposed Google and Cohere key in the respective dashboards.
2. Review provider usage and billing for unexpected activity.
3. Configure replacement keys outside version control, using environment variables
   or the password field in the local app.
4. If repository-history cleanup is desired, coordinate it with collaborators:
   rewriting shared history changes commit IDs and requires fresh clones/rebases.
   It cannot remove copies in forks, caches, or previous downloads and is not a
   substitute for revoking credentials.

No history rewrite is performed by this change. Until the pull request is merged,
the default branch still contains the old application. Do not use its embedded keys.

The application never executes model-generated code. Uploaded filenames are not
used as disk paths, indexes are session-local, and raw provider errors are not
shown. These improvements do not make the app a hardened public service. Add
authentication, per-user quotas, stronger resource isolation, monitoring, and
deployment-specific controls before exposing it to untrusted users.
