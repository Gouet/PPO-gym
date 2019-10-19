call C:\Users\Victor\Anaconda3\Scripts\activate.bat
call conda activate GYM_ENV_RL

set scenario=LunarLander-v2

python train.py --scenario=LunarLander-v2
pause