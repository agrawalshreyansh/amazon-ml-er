# ponytail: pickle shim only. Models saved by the old single-file version are pickled as er_fast.Model;
# this lets models/*.pkl from an earlier run still load. Delete once no old work dir is reused.
from stage4_train import Model  # noqa: F401
