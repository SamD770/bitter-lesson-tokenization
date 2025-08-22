
base_dir="/scratch_net/tikgpu07/sdauncey/tokenizer_training/fineweb_100B_filtered"
to_dir="/scratch/sdauncey/tokenizer_training/fineweb_100B_filtered"

rsync -av --progress $base_dir/ $to_dir/