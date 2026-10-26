# Component extraction

This repository packages the existing `src/auv_marine_sim` component of the Agentic AUV project as an independent Python simulator.

The source commit and original source-file checksums are recorded in [extraction_source.json](extraction_source.json). The simulator physics, fault models, sensor models, referee and automatic-label implementation were copied from that component. Packaging changes make YAML configuration load from this installed package instead of a ROS overlay or the old workspace.

The standalone layer adds the CLI, examples, packaged resource metadata, installation checks, CI configuration and passive fault visualizations. It preserves the `auv_marine_sim` Python import name so integrations can continue to use the simulator API.

The integrated Agentic AUV repository remains responsible for mission agents, navigation, perception models, deterministic execution/control and ROS messages. A simulator-library installation does not imply those applications or their model weights are installed.

No repository history is copied into this directory. The owner initializes and pushes the standalone repository after reviewing the prepared files.
