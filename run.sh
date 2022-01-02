#!/bin/zsh
#python main_mao.py --impedance_mode=variable --env_name=Door
#python main_mao.py --impedance_mode=fixed --env_name=Door
#python main_mao.py --impedance_mode=variable_kp --env_name=Door

python main_mao2.py --impedance_mode="fixed" --seed 88 --env="Door" --robots="Panda" --agent="SAC" --variant="/home/sumail/robosuite/params/params/variant.json"