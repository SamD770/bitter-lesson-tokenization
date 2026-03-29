CHECKPOINT_PATH=$1

python -m eval.run --checkpoint $CHECKPOINT_PATH --batch-size 1
python -m eval.fineweb_test --checkpoint $CHECKPOINT_PATH --batch-size 1 --limit 4000