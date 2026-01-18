# How to Update the Streamlit App Deployment

## Quick Update Steps

After pushing changes to GitHub, follow these steps:

### Step 1: SSH into the EC2 instance

```bash
# Replace with your actual key path and user
ssh -i ~/.ssh/your-key.pem ubuntu@54.89.18.194
```

### Step 2: Find where the app is located

Try these common locations:

```bash
# Check home directory
ls -la ~/streamlit_app_deploy
ls -la ~/aladynoulli2

# Check common web directories
ls -la /var/www/
ls -la /opt/

# Search for the app file
find ~ -name "patient_timeline_app_compact.py" 2>/dev/null
find /var/www -name "patient_timeline_app_compact.py" 2>/dev/null
```

### Step 3: Navigate to the app directory and pull changes

```bash
# Once you find it, navigate there
cd /path/to/app/directory

# Pull latest changes
git pull origin main

# If not a git repo, you may need to:
# 1. Clone it fresh, OR
# 2. Manually update files via scp (see below)
```

### Step 4: Restart the Streamlit service

The service could be running in several ways. Try each:

#### Option A: Systemd service
```bash
# Check if there's a systemd service
sudo systemctl status streamlit
sudo systemctl status streamlit-app
sudo systemctl list-units | grep streamlit

# If found, restart it:
sudo systemctl restart streamlit
# OR
sudo systemctl restart streamlit-app
```

#### Option B: Screen/Tmux session
```bash
# List screen sessions
screen -ls

# List tmux sessions
tmux ls

# If found, attach and restart manually
screen -r <session-name>
# Then Ctrl+C and restart the app
```

#### Option C: Docker/Container
```bash
# Check for Docker containers
docker ps
docker-compose ps

# Restart if found
docker-compose restart
# OR
docker restart <container-name>
```

#### Option D: Process running directly
```bash
# Find the Streamlit process
ps aux | grep streamlit

# Kill it (note the PID from above)
kill <PID>

# Restart (you'll need to know the command)
# Usually something like:
cd /path/to/app
streamlit run patient_timeline_app_compact.py --server.port=8501 &
```

---

## Alternative: Manual Update (if Git pull doesn't work)

If the server doesn't have git set up, you can update files manually:

### From your local machine:

```bash
# Navigate to the app directory
cd /Users/sarahurbut/aladynoulli2/streamlit_app_deploy

# Copy updated files to EC2
scp -i ~/.ssh/your-key.pem patient_timeline_app_compact.py ubuntu@54.89.18.194:/path/to/app/
scp -i ~/.ssh/your-key.pem disease_names.csv ubuntu@54.89.18.194:/path/to/app/
# Add other files if needed

# Then SSH in and restart the service (see Step 4 above)
```

---

## Troubleshooting: Find Out What's Running

If you're not sure how it's set up, run these commands on the EC2 instance:

```bash
# 1. Check what processes are running
ps aux | grep streamlit
ps aux | grep python

# 2. Check what ports are listening
sudo netstat -tlnp | grep 8501
sudo ss -tlnp | grep 8501

# 3. Check systemd services
systemctl list-units --type=service | grep streamlit

# 4. Check for Docker
docker ps -a

# 5. Check nginx configuration (to see where it's proxying)
sudo cat /etc/nginx/sites-available/default
sudo cat /etc/nginx/sites-enabled/*

# 6. Check for systemd service files
ls -la /etc/systemd/system/*streamlit*
```

---

## Contact Your Friend

If you can't figure it out, ask your friend for:
1. The exact path where the app is located on EC2
2. How Streamlit is running (systemd, screen, Docker, etc.)
3. The SSH key file location or access instructions
4. Any specific commands they use to restart it

---

## Future: Set Up Automated Deployment

Once you know how it's set up, you could:
1. Create a simple update script (see `update_app.sh` below)
2. Set up GitHub Actions to auto-deploy
3. Use a deployment tool like Ansible

