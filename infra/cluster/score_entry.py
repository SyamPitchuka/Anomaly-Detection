"""spark-submit entry point for src/streaming/score_job.py on the standalone cluster (used by submit.sh).

score_job.py hard-codes the Kafka bootstrap (contracts.schemas.KAFKA_BOOTSTRAP = localhost:9092) and its
checkpoint directory (<repo>/checkpoints/score/source=smd). Both are replaced IN MEMORY before score_job.main()
runs, exactly as run_job.py does, so score_job.py itself is not edited:

    CLUSTER_KAFKA_BOOTSTRAP   full broker list, e.g. spark-master:9092   -> contracts.schemas.KAFKA_BOOTSTRAP
    SCORE_CHECKPOINT_DIR      checkpoint directory on the shared storage -> score_job.CHECKPOINT_DIR

Everything after the script name goes to score_job's own argument parser (--input, --max-seconds, ...).
The Spark master is NOT set here: submit.sh passes --master <spark://...> to score_job, because score_job calls
SparkSession.builder.master(args.master) (default local[*]) and that call would override spark-submit's --master.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from contracts import schemas as S
    from src.streaming import score_job

    bootstrap = os.environ.get("CLUSTER_KAFKA_BOOTSTRAP")
    if bootstrap:
        S.KAFKA_BOOTSTRAP = bootstrap
    checkpoint = os.environ.get("SCORE_CHECKPOINT_DIR")
    if checkpoint:
        score_job.CHECKPOINT_DIR = Path(checkpoint)
    print(f"score_entry: Kafka {S.KAFKA_BOOTSTRAP}, checkpoint {score_job.CHECKPOINT_DIR.as_posix()}", flush=True)
    return score_job.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
