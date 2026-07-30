# Configuration

Versioned, human-readable site and experiment inputs live here.

Configuration files declare intent. A production campaign must resolve them
into an immutable, content-hashed `campaign_spec.json`; changing a YAML file
after submission must not alter an active campaign.

`tigris.yaml` records site defaults. Resource requests remain measured
campaign choices rather than universal constants.
