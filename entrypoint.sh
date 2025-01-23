#!/bin/bash
set -e

# /bin/bash -c "python3 /home/$user/src/generate_synthetic_data/manage.py makemigrations"
# /bin/bash -c "python3 /home/$user/src/generate_synthetic_data/manage.py migrate"
# /bin/bash -c "python3 /home/$user/src/generate_synthetic_data/manage.py create_superuser"

sudo -E supervisord -n -c /etc/supervisord.conf