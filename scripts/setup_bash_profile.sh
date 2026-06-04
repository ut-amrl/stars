#!/bin/bash
# Script to combine bash_profile files
# This handles the optional bash_profile.local file

USERNAME=$1

# Start with shared configuration
cp /tmp/bash_profile.shared /home/${USERNAME}/.bash_profile

# Replace placeholders with actual values
sed -i "s|__USERNAME__|${USERNAME}|g" /home/${USERNAME}/.bash_profile

chown ${USERNAME}:${GID} /home/${USERNAME}/.bash_profile || true
chmod 644 /home/${USERNAME}/.bash_profile

