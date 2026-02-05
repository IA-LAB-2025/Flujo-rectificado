# Flujo-rectificado
Proyecto de investigación en inteligencia artificial sobre el modelado de flujo rectificado y la generación de imágenes mediante técnicas de deep learning.

comando tree:
´tree -L 2 -I 'data|__pycache__|*.jpg|*.png|*.jpeg|*.npy|*.csv'´

entrenamientos:
´python train.py --limit 1000 --batch_size 32 --epochs 100 --channels 128 --exp_name cheetah_v2_full´
