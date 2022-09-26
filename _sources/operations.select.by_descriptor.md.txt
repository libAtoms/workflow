# Selection or Sampling of Structures

Training of machine learning potentials often requires picking-out a few structures from a large configurations-database (for example, an MD trajectory). This can be achieved in a few different ways in the Workflow package depending on one's choice of the selection criteria (see `wfl.select`).

Selection of a set of individually unique structures can be done by comparing descriptors for each configuration in the database (see `wfl.select.by_descriptor`). Here, you can find functions to process descriptors as well as functions to perform two different selection algorithms, namely leverage-score CUR and greedy farthest-point-first (FPS). Additional features including exclusion of a list of structures or consideration of previously selected structures can be passed as arguments in the selection criteria.

`example.select_fps` shows an example of FPS selection from an MD trajectory by comparison of average SOAP descriptors.
