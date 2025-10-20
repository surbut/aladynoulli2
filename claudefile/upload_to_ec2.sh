#!/bin/bash
# Upload aladyn_project to EC2
# Usage: bash upload_to_ec2.sh <EC2-IP> <path-to-key.pem>

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash upload_to_ec2.sh <EC2-IP> <path-to-key.pem>"
    echo "Example: bash upload_to_ec2.sh 54.123.45.67 ~/.ssh/my-key.pem"
    exit 1
fi

EC2_IP=$1
KEY_FILE=$2

echo "Uploading aladyn_project to EC2 instance at $EC2_IP..."

# Upload the entire aladyn_project directory
scp -i "$KEY_FILE" -r aladyn_project ubuntu@$EC2_IP:~/

echo ""
echo "Upload complete!"
echo ""
echo "Next steps:"
echo "1. SSH into EC2: ssh -i $KEY_FILE ubuntu@$EC2_IP"
echo "2. Run setup: bash ~/aladyn_project/ec2_setup.sh"
echo "3. Logout and login again"
echo "4. Run predictions: bash ~/aladyn_project/ec2_run_predictions.sh"
