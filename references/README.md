# Locked design references

This folder preserves the reviewed 7.16 metric/state source tables used to create the configuration registry. They are evidence references, not runtime inputs. Runtime behavior is controlled only by versioned YAML under `configs/`, which avoids silently reading a changed spreadsheet during model execution.

The migration rule is explicit: a spreadsheet revision is reviewed, converted to YAML, tested, and committed. It is never loaded directly into a production run.

