  # First, make sure you're in the right directory and have your conda env activated
  cd /Users/sarahurbut/aladynoulli2/claudefile
  conda activate new_env_pyro2

  # Create logs directory if it doesn't exist
  mkdir -p logs

  # Run the prediction in the background with nohup
  nohup python run_aladyn_predict.py \
      --trained_model_path /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_model_W0.0001_fulldata_sexspecific.pt \
      --data_dir /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/ \
      --output_dir output/ \
      --covariates_path /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/baselinagefamh_withpcs.csv \
      --batch_size 10000 \
      --num_epochs 200 \
      --learning_rate 0.1 \
      --lambda_reg 0.01 \
      > logs/predict.log 2>&1 &

  # This will print the process ID (PID)
  echo "Prediction started with PID: $!"

  To monitor progress:
  tail -f logs/predict.log

  To check if it's still running:
  ps aux | grep run_aladyn_predict

  To stop it if needed:
  pkill -f run_aladyn_predict

