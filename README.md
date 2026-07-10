```markdown
# MnemonicAi — Consolidated Box

A complete guide to installing, running, and maintaining your AI memory system.

---

## Table of Contents

1. [What Is MnemonicAi?](#1-what-is-mnemonicai)
2. [What You Need Before You Start](#2-what-you-need-before-you-start)
3. [Plain-English Glossary](#3-plain-english-glossary)
4. [Installation (Bare Metal)](#4-installation-bare-metal)
5. [Installation (Docker)](#5-installation-docker)
6. [Starting and Stopping the Server](#6-starting-and-stopping-the-server)
7. [Training the Model (QLoRA Bakes)](#7-training-the-model-qlora-bakes)
8. [Adapter Versioning and Rollback](#8-adapter-versioning-and-rollback)
9. [Monitoring with Grafana and Prometheus](#9-monitoring-with-grafana-and-prometheus)
10. [Backups and Recovery](#10-backups-and-recovery)
11. [Systemd (Auto-Start on Boot)](#11-systemd-auto-start-on-boot)
12. [Docker Image CI/CD Pipeline](#12-docker-image-cicd-pipeline)
13. [Kubernetes (Fleet Deployment)](#13-kubernetes-fleet-deployment)
14. [Configuration Reference](#14-configuration-reference)
15. [Troubleshooting](#15-troubleshooting)
16. [Script Reference](#16-script-reference)

---

## 1. What Is MnemonicAi?

MnemonicAi is a **memory-native AI system** — it runs a language model on your own hardware, learns from conversations, and remembers what it learned across sessions. Unlike ChatGPT or Claude (which run on someone else's servers and forget everything when you close the tab), MnemonicAi runs entirely on your machine and builds a growing memory of your interactions.

### Your hardware setup

| Component | What it does | Role in MnemonicAi |
|-----------|-------------|-------------------|
| AMD Ryzen 9 5950X | The CPU (16 cores) | Runs the Python code, manages data, coordinates everything |
| 64 GB DDR4 RAM | System memory | Holds the operating system, Python, and model data while running |
| RTX 4080 SUPER | GPU 0 (graphics card) | **Inference** — generates responses when you talk to the AI |
| RTX 3090 | GPU 1 (graphics card) | **Training** — learns from new data in the background |
| Ubuntu 24.04 LTS | The operating system | Everything runs on Linux |

### Two GPUs, two jobs

The key idea: your system has two graphics cards, and MnemonicAi assigns each one a specific job:

- **GPU 0 (RTX 4080 SUPER)** handles **inference** — this means generating text when you ask the AI something. It needs to be fast and responsive.
- **GPU 1 (RTX 3090)** handles **training** — this means learning from new data. It's slower and runs in the background.

This split means you can chat with the AI (using GPU 0) while it's simultaneously learning from new material (using GPU 1), without either task slowing down the other.

---

## 2. What You Need Before You Start

### Hardware

- The computer described above (or similar — at least 1 NVIDIA GPU with 24+ GB VRAM)
- A monitor and keyboard (or SSH access)
- At least 100 GB of free disk space (for the model files and training data)

### Software you need installed first

If any of these are missing, the install script will try to install them for you, but it's best to have them ready:

| Software | How to check | How to install if missing |
|----------|-------------|--------------------------|
| Ubuntu 22.04 or 24.04 | `cat /etc/os-release` | It's your operating system — you already have it |
| Git | `git --version` | `sudo apt install git` |
| Curl | `curl --version` | `sudo apt install curl` |
| Internet connection | `ping github.com` | Make sure you're online |

### Files you need

You need the MnemonicAi repository on your computer. If you haven't downloaded it yet:

```bash
cd ~
git clone https://github.com/YOUR_ORG/MnemonicAi.git
cd MnemonicAi
```

You also need the model files (the actual AI brain). These should be in:

```
MnemonicAi/
  models/
    ornith-1.0-9b/           ← the raw model files (safetensors)
    ornith-1.0-9bgguf/       ← the compressed (GGUF) version
```

If these folders are empty, you need to download the model weights first. Ask whoever set up your project where to get them.

---

## 3. Plain-English Glossary

Don't skip this section — understanding these terms makes everything else make sense.

| Term | What it means in plain English |
|------|-------------------------------|
| **Inference** | The AI generating a response when you ask it something. Like when ChatGPT writes an answer — that's inference. |
| **Training** | The AI learning from new data. Each "training run" or "bake" teaches the AI something new. |
| **QLoRA** | A specific training method that's very memory-efficient. It doesn't change the whole model — it creates a small "adapter" (like a patch) that modifies behavior. |
| **Adapter** | A small file (50-200 MB) that contains what the AI learned in a training run. Think of it like a sticky note the AI reads before answering — it changes behavior without rewriting the whole brain. |
| **GGUF** | A compressed format for model files. It makes the model smaller so it loads faster and uses less memory. |
| **GPU** | Your graphics card. AI uses it because it's very good at the math AI needs. |
| **VRAM** | Memory on your graphics card. The RTX 3090 has 24 GB; the 4080 SUPER has 16 GB. |
| **CUDA** | Software from NVIDIA that lets Python use your graphics card. |
| **PyTorch** | The Python library that does the actual AI math. |
| **Docker** | A way to package the entire system into a container that runs the same on any computer. |
| **Kubernetes (K8s)** | A system for managing many containers across many computers. Overkill for one machine; useful when you have a fleet. |
| **Prometheus** | Software that collects metrics (numbers about how your system is performing). |
| **Grafana** | Software that draws dashboards and charts from Prometheus data. |
| **Systemd** | The system that manages services on Ubuntu. It can auto-start MnemonicAi when you boot up. |
| **Adapter versioning** | Keeping track of which adapter (learning patch) is active, so you can switch between them or roll back if one makes the AI worse. |
| **OOM** | "Out of Memory" — when the graphics card runs out of VRAM. The most common training error. |

---

## 4. Installation (Bare Metal)

This installs MnemonicAi directly on your computer without Docker. Choose this if you're running on a single machine.

### Step 1: Open a terminal

Press `Ctrl + Alt + T` on Ubuntu, or SSH into your machine.

### Step 2: Go to the project folder

```bash
cd ~/MnemonicAi
```

If you get an error saying "no such directory," the folder isn't there. See [Section 2](#2-what-you-need-before-you-start) to download it first.

### Step 3: Make the scripts executable

```bash
chmod +x mn_*.sh
```

This tells Ubuntu "these files are programs, not just text." You only need to do this once.

### Step 4: Run the installer

```bash
./mn_install.sh
```

**What happens during installation:**

1. It checks your operating system version
2. It installs Python 3.12 (the programming language MnemonicAi uses)
3. It syncs the latest code from Git
4. It creates a virtual environment (an isolated Python setup so dependencies don't conflict)
5. It installs all Python libraries (PyTorch, Transformers, etc.)
6. It checks for your NVIDIA graphics cards and CUDA drivers
7. It installs the NVIDIA driver and CUDA toolkit (if not already present)
8. It installs PyTorch with CUDA support (so it can use your GPUs)
9. It runs a smoke test to verify both GPUs work
10. It checks your PCIe slot configuration (warns if a GPU is in a slow slot)
11. It verifies the model files are present
12. It writes a configuration file and assigns GPU roles

**Important:** The installer will **reboot your computer** after installing the NVIDIA driver (step 7). This is normal. After the reboot, run the installer again — it will skip everything that's already done and continue from where it left off.

If you don't want it to reboot automatically:

```bash
MN_SKIP_REBOOT=1 ./mn_install.sh
```

You'll need to reboot manually later for the NVIDIA driver to take effect.

### Step 5: After the reboot, run the installer again

```bash
cd ~/MnemonicAi
./mn_install.sh
```

It should say "already done" for most steps and complete quickly. When you see the summary at the end showing "OK" counts and no critical failures, you're ready.

### Step 6: Verify everything is working

```bash
./mn_run.sh status
```

You should see something like:

```
=== process ===
NOT RUNNING

=== GPUs ===
0  NVIDIA GeForce RTX 4080 SUPER  0%  200MiB/16376MiB  35C
1  NVIDIA GeForce RTX 3090        0%  300MiB/24576MiB  40C
```

If you see both GPUs listed, installation is complete. If not, see [Troubleshooting](#15-troubleshooting).

---

## 5. Installation (Docker)

Docker packages everything into a container — like a box that contains the entire system. This is recommended if you want to run MnemonicAi on multiple machines or want reproducible setups.

### Prerequisites

You need Docker and the NVIDIA Container Toolkit installed first:

```bash
# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose

# Add yourself to the docker group so you don't need sudo
sudo usermod -aG docker $USER
# Log out and log back in for this to take effect

# Install NVIDIA Container Toolkit (lets Docker use your GPUs)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build and start

```bash
cd ~/MnemonicAi
docker compose up --build -d
```

The first build takes 15-30 minutes (it downloads PyTorch, CUDA libraries, etc.). Subsequent rebuilds are much faster thanks to caching.

### Verify

```bash
# Check the container is running
docker compose ps

# Check the logs
docker compose logs -f mnemonicai

# Test the health endpoint
curl http://localhost:8400/health
```

You should see: `{"status": "healthy"}`

### What's running

| URL | What it does |
|-----|-------------|
| `http://localhost:8400` | The main API (inference server) |
| `http://localhost:8400/v1` | OpenAI-compatible API endpoint |
| `http://localhost:8400/events` | Server-sent events stream (live updates) |
| `http://localhost:8401/admin/adapters` | Adapter management web UI |

### Stopping Docker

```bash
docker compose down
```

---

## 6. Starting and Stopping the Server

### Start the server (runs in the background)

```bash
./mn_run.sh background
```

This starts the inference server on GPU 0 (your RTX 4080 SUPER). It will keep running even if you close your terminal.

### Start the server (foreground, for debugging)

```bash
./mn_run.sh
```

or

```bash
./mn_run.sh serve
```

This shows the logs directly in your terminal. Press `Ctrl + C` to stop.

### Check if the server is running

```bash
./mn_run.sh status
```

Shows whether the server is running, which GPU it's using, how much memory is consumed, and the last few log lines.

### View logs

```bash
# Last 100 log lines
./mn_run.sh logs

# Last 500 log lines
./mn_run.sh logs 500
```

### Stop the server

```bash
./mn_run.sh stop
```

Sends a graceful shutdown signal (SIGTERM), waits up to 10 seconds, then force-kills if needed.

### Start with a mock backend (no GPU needed, for testing)

```bash
./mn_run.sh mock
```

This runs the server without loading the model onto the GPU. Useful for testing the API without using GPU resources.

### View the current configuration

```bash
./mn_run.sh env
```

Shows the contents of `mn_env.json` (GPU info), the GPU role assignment, and runtime settings.

---

## 7. Training the Model (QLoRA Bakes)

"Training" or "baking" is how the AI learns from new data. Each training run creates an **adapter** — a small file that modifies the AI's behavior based on what it learned.

### How to run a training bake

```bash
./mn_run.sh train
```

**What happens:**

1. The system loads the base model (ornith-1.0-9b) onto GPU 1 (your RTX 3090)
2. It gathers examples from the memory database
3. It runs QLoRA training (uses 4-bit quantization + LoRA adapters — very memory-efficient)
4. It saves the adapter to `mnemonicai_data/adapter/vN/`
5. It evaluates the new adapter against held-out data
6. If the evaluation is worse than the current adapter, it rolls back automatically

**What you'll see:**

```
[14:32:01] QLoRA training run. Device = GPU 1. Adapter -> mnemonicai_data/adapter
Loading model in 4-bit...
GPU memory: 14.2 GB / 24.0 GB
Training:   0%|          | 0/8 [00:00<00:00,  0.00it/s]
Training:  12%|█▎        | 1/8 [00:12<01:24,  0.09it/s, loss=2.145]
...
Training: 100%|██████████| 8/8 [02:15<00:00,  0.06it/s, loss=1.823]
Eval loss: 1.912
Adapter saved to mnemonicai_data/adapter/v3/
TRAIN_STACK_OK
```

### How long does training take?

- A typical 8-step bake takes **2-3 minutes** on the RTX 3090
- The time depends on how many examples are in the memory database
- The GPU gets hot during training (~70-85°C is normal)

### If you get an "OOM" (Out of Memory) error

The most common training error. The RTX 3090 has 24 GB of VRAM, and the training process can exceed it. Fix it by editing `config.json`:

```json
{
  "qlora_batch_size": 2
}
```

Change `4` to `2` (or even `1`). This halves the memory usage at the cost of slightly slower training. See [Configuration Reference](#14-configuration-reference) for all options.

### How to wire the adapter registry into your training code

After each successful bake, your training code needs to register the new adapter so it shows up in the version manager. Add this function to your `SleepTrainer` or wherever the adapter is saved:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

def register_adapter(adapter_dir: Path, version: str, metrics: dict):
    """Call this after each training run to register the new adapter."""
    registry = Path(adapter_dir) / ".registry.json"
    data = {"versions": [], "active": None, "previous": None}
    if registry.exists():
        data = json.loads(registry.read_text())
    entry = {
        "name": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_loss": metrics.get("loss"),
        "eval_loss": metrics.get("eval_loss"),
        "examples": metrics.get("examples"),
        "steps": metrics.get("steps"),
    }
    data["versions"].append(entry)
    data["active"] = version
    data["previous"] = data.get("active") if data.get("active") != version else data.get("previous")
    registry.write_text(json.dumps(data, indent=2))
```

Call it after training completes:

```python
register_adapter(
    adapter_dir="mnemonicai_data/adapter",
    version="v3",
    metrics={"loss": 1.823, "eval_loss": 1.912, "examples": 12, "steps": 8}
)
```

---

## 8. Adapter Versioning and Rollback

Every time the AI trains, it creates a new **adapter** (a learning patch). The system keeps track of all adapters so you can:

- See which one is currently active
- Compare two versions to see which performs better
- Roll back to a previous version if the new one makes things worse
- Delete old versions to save disk space

### Using the command-line tool

```bash
# List all adapters with their metrics
./mn_adapter.sh list

# Activate a specific version (switches the active adapter)
./mn_adapter.sh activate v3

# Roll back to the previous version (undo the last activation)
./mn_adapter.sh rollback

# Compare two versions side by side
./mn_adapter.sh compare v2 v3

# Delete old versions, keep the 5 most recent
./mn_adapter.sh gc 5
```

**Example output of `list`:**

```
VERSION       ACTIVE    TRAIN_LOSS   EVAL_LOSS    TIMESTAMP
v1                      2.145        2.312        2026-01-15T10:30:00Z
v2                      1.987        2.201        2026-01-15T11:15:00Z
v3            =>        1.823        1.912        2026-01-15T12:00:00Z
```

The `=>` shows which version is currently active.

### Using the web UI

If you started the adapter UI (it starts automatically with Docker, or run `./mn_run.sh adapter-ui`):

1. Open `http://localhost:8401/admin/adapters` in your browser
2. You'll see a table of all adapter versions with their training metrics
3. Click **Activate** next to any version to make it the active one
4. Click **Rollback** to undo the last activation
5. Click **Garbage Collect** to delete old versions

### When to roll back

Roll back if:
- The AI's responses got worse after a training run
- The evaluation loss went up instead of down
- The AI started producing incorrect or nonsensical output

Rolling back is instant — it just switches which adapter file the system reads. No retraining needed.

---

## 9. Monitoring with Grafana and Prometheus

Prometheus collects metrics (numbers about how your system is performing). Grafana displays those numbers as charts and dashboards. Together, they let you see exactly what's happening in real time.

### Setting up the monitoring stack with Docker

#### Step 1: Add the Prometheus library to your dependencies

```bash
echo "prometheus-fastapi-instrumentator>=7.0" >> requirements.txt
```

#### Step 2: Start the full stack

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d --build
```

This starts:
- MnemonicAi (your AI server)
- Prometheus (metrics collector)
- Grafana (dashboards)
- Node Exporter (system metrics like CPU, RAM, disk)

#### Step 3: Verify metrics are flowing

```bash
curl http://localhost:8400/metrics | head -20
```

You should see lines like:

```
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{handler="/",method="GET",status="2xx"} 42.0
...
```

If you see this, Prometheus is collecting data successfully.

#### Step 4: Open Grafana

Open your browser and go to: `http://localhost:3000`

- **Username:** `admin`
- **Password:** `admin` (or whatever you set in your `.env` file)

On first login, Grafana asks you to change the password. Change it to something secure.

#### Step 5: View the dashboard

1. Click **Dashboards** in the left sidebar
2. You should see **"MnemonicAi — Inference & Training"**
3. Click it to open a 12-panel dashboard showing:
   - Request rate (how many requests per second)
   - Error rate (percentage of failed requests)
   - P95 latency (95% of requests complete within this time)
   - Active adapter version (which learning patch is active)
   - Training loss and evaluation loss over time
   - Training events (successes, failures, rollbacks)
   - GPU memory usage and utilization
   - Overall health status

#### Step 6: Open Prometheus (optional, for advanced queries)

Go to `http://localhost:9090`

- Click **Status → Targets** to see if MnemonicAi is being scraped
- The `mnemonicai` job should show as **UP**
- You can type queries in the search bar, e.g.:
  - `rate(http_requests_total[5m])` — requests per second over the last 5 minutes
  - `mnemonicai_training_loss` — current training loss
  - `mnemonicai_gpu_memory_used_bytes / mnemonicai_gpu_memory_total_bytes` — GPU memory fraction used

### Setting up Google sign-in for Grafana

If you don't want to use the default `admin/admin` login, you can set up Google OAuth so anyone in your Google Workspace can sign in with their Google account.

#### Step 1: Create Google OAuth credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**
4. Application type: **Web application**
5. Add authorized redirect URI: `http://localhost:3000/login/google`
6. Copy the **Client ID** and **Client Secret**

#### Step 2: Create a `.env` file

Create a file called `.env` in your project root (next to `docker-compose.yml`):

```bash
# .env — DO NOT COMMIT THIS FILE (it should be in .gitignore)
GRAFANA_PASSWORD=your-strong-random-password-here
GRAFANA_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GRAFANA_GOOGLE_CLIENT_SECRET=GOCSPX-your-secret
GRAFANA_HOSTED_DOMAIN=yourdomain.com
```

Generate a strong password:

```bash
openssl rand -base64 24
```

#### Step 3: Make sure `.env` is not committed

```bash
echo ".env" >> .gitignore
```

#### Step 4: Add Google OAuth settings to docker-compose.observability.yml

In the `grafana:` service, add these environment variables:

```yaml
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false

      # Google OAuth
      - GF_AUTH_GOOGLE_ENABLED=true
      - GF_AUTH_GOOGLE_CLIENT_ID=${GRAFANA_GOOGLE_CLIENT_ID}
      - GF_AUTH_GOOGLE_CLIENT_SECRET=${GRAFANA_GOOGLE_CLIENT_SECRET}
      - GF_AUTH_GOOGLE_SCOPES=https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email
      - GF_AUTH_GOOGLE_AUTH_URL=https://accounts.google.com/o/oauth2/auth
      - GF_AUTH_GOOGLE_TOKEN_URL=https://accounts.google.com/o/oauth2/token
      - GF_AUTH_GOOGLE_API_URL=https://www.googleapis.com/oauth2/v1/userinfo
      - GF_AUTH_GOOGLE_ALLOW_SIGN_UP=true
      - GF_AUTH_GOOGLE_HOSTED_DOMAIN=${GRAFANA_HOSTED_DOMAIN:-yourdomain.com}

      # Optional: hide the local login form entirely (force Google sign-in)
      # - GF_AUTH_DISABLE_LOGIN_FORM=true
      # - GF_AUTH_GOOGLE_AUTO_LOGIN=true
```

#### Step 5: Restart Grafana

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml down
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
```

Now when you go to `http://localhost:3000`, you'll see a "Sign in with Google" button. The local `admin` account still exists as a fallback (with the password you set in `.env`).

### Alert rules

The system includes 6 pre-configured alerts in `prometheus/alerts.yml`:

| Alert | What it detects | Severity |
|-------|----------------|----------|
| MnemonicAiDown | Server unreachable for 1+ minutes | Critical |
| MnemonicAiHighLatency | P95 latency > 5 seconds for 5+ minutes | Warning |
| MnemonicAiHighErrorRate | Error rate > 10% for 2+ minutes | Warning |
| MnemonicAiTrainingFailure | A training run failed in the last hour | Warning |
| MnemonicAiGPUMemoryHigh | GPU memory > 95% used for 5+ minutes | Warning |
| MnemonicAiAdapterRollback | A rollback was triggered in the last hour | Info |

---

## 10. Backups and Recovery

Backups protect your data (memories, adapters, conversation history) in case of disk failure or accidental deletion.

### Run a backup manually

```bash
./mn_backup.sh
```

**What it does:**

1. If the server is running, it takes a consistent SQLite snapshot (using `VACUUM INTO` — this means the backup won't be corrupted by ongoing writes)
2. If the server is not running, it copies the data directory directly
3. It creates a compressed archive (`.tar.zst` — very efficient compression)
4. It computes a SHA-256 checksum (so you can verify the backup isn't corrupted)
5. If you've configured an offsite destination (like a NAS or USB drive), it copies the backup there
6. It prunes old backups according to the retention policy:
   - Keeps 7 daily backups
   - Keeps 4 weekly backups
   - Keeps 6 monthly backups

### Set up automatic daily backups (cron)

Cron is a Linux tool that runs things on a schedule. To run the backup every day at 3:17 AM:

```bash
crontab -e
```

Add this line at the end:

```
17 3 * * * /home/$USER/MnemonicAi/mn_backup.sh >> /home/$USER/MnemonicAi/mnemonicai_data/logs/backup.log 2>&1
```

Save and exit. Backups will now run automatically every night.

### Restoring from a backup

```bash
# 1. Stop the server
./mn_run.sh stop

# 2. Find your backup
ls mnemonicai_data/backups/

# 3. Extract the backup
mkdir -p /tmp/restore
tar --use-compress-program="zstd -d" -xf mnemonicai_data/backups/mn_hostname_20260115T031700Z.tar.zst -C /tmp/restore

# 4. Verify the checksum
sha256sum mnemonicai_data/backups/mn_hostname_20260115T031700Z.tar.zst

# 5. Replace the current data
rm -rf mnemonicai_data/memory.db
cp /tmp/restore/memory.db mnemonicai_data/
cp -r /tmp/restore/adapter mnemonicai_data/

# 6. Restart
./mn_run.sh background
```

### Configuring offsite backups

Edit `mn_lib.sh` and set:

```bash
export MN_BACKUP_OFFSITE="/mnt/nas/mnemonicai_backups"
```

Point this to a mounted NAS drive, USB drive, or any directory on another disk. The backup script will copy the archive there after creating it locally.

---

## 11. Systemd (Auto-Start on Boot)

Systemd is the system manager on Ubuntu. It can start MnemonicAi automatically when you boot up, and monitor it for crashes.

### Install the systemd service

```bash
sudo ./mn_service.sh
```

**What this does:**

1. Creates a systemd service file at `/etc/systemd/system/mnemonicai.service`
2. Creates a health-check timer that runs every 5 minutes
3. Enables both so they start on boot
4. Starts the service immediately

**Important design choice:** The service is set to `Restart=no`. This means if the AI crashes, it stays down. This is intentional — you want to know when it crashes, not have it silently restart and hide the problem. The health timer sends you an alert (via email or webhook) instead.

### Managing the service

```bash
# Check status
systemctl status mnemonicai

# Start manually
sudo systemctl start mnemonicai

# Stop
sudo systemctl stop mnemonicai

# Restart
sudo systemctl restart mnemonicai

# View logs
journalctl -u mnemonicai -f

# Check the health timer
systemctl list-timers mn-watch.timer
```

### Configuring alerts

The health checker can send alerts via email or webhook (Slack, Discord, etc.) when the server goes down. Edit `mn_lib.sh`:

```bash
export MN_ALERT_EMAIL="you@yourdomain.com"
export MN_ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

If both are blank, no alerts are sent. If you set at least one, the health checker will use it when the server is down.

---

## 12. Docker Image CI/CD Pipeline

If you're using GitHub, the included workflows automatically build and push your Docker image whenever you push code.

### How it works

When you push to the `main` branch:

1. GitHub Actions builds the Docker image
2. It pushes the image to GitHub Container Registry (`ghcr.io`)
3. It tags the image with:
   - `latest` (always points to the newest build on main)
   - `sha-abc1234` (the git commit SHA — immutable, for rollback)
   - `main` (the branch name)
4. It runs a smoke test (starts the container and checks the health endpoint)
5. It generates an SBOM (Software Bill of Materials — a list of all dependencies for security auditing)

When you push a tag like `v1.2.0`:

1. The same build happens
2. The image gets tagged as `v1.2.0`
3. A GitHub Release is automatically created with release notes

### Using the CI/CD-built image

After the workflow runs, anyone can pull your image:

```bash
docker pull ghcr.io/YOUR_GITHUB_ORG/mnemonicai:latest
docker run --gpus all -p 8400:8400 -v $(pwd)/mnemonicai_data:/app/mnemonicai_data ghcr.io/YOUR_GITHUB_ORG/mnemonicai:latest
```

### Creating a release

```bash
git tag v1.0.0
git push origin v1.0.0
```

This triggers the release workflow. The GitHub Release page will have release notes and the Docker image will be available as `ghcr.io/YOUR_GITHUB_ORG/mnemonicai:v1.0.0`.

### Rolling back to a previous version

```bash
# Pull a specific version
docker pull ghcr.io/YOUR_GITHUB_ORG/mnemonicai:sha-abc1234

# Run it instead of latest
docker run --gpus all -p 8400:8400 ghcr.io/YOUR_GITHUB_ORG/mnemonicai:sha-abc1234
```

**Never use `latest` in production.** Always pin to a specific version tag or SHA so you know exactly what's running.

---

## 13. Kubernetes (Fleet Deployment)

Kubernetes is for when you have multiple computers and want to run MnemonicAi across all of them. If you only have one machine, use Docker Compose instead (see [Section 5](#5-installation-docker)).

### Prerequisites

On every GPU node in your cluster:

1. NVIDIA drivers installed
2. NVIDIA Container Toolkit configured
3. NVIDIA device plugin deployed:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
```

### Deploy MnemonicAi

```bash
# 1. Replace YOUR_GITHUB_ORG in the K8s manifests
sed -i 's/YOUR_GITHUB_ORG/your-actual-org/g' k8s/deployment.yaml k8s/train-job.yaml k8s/kustomization.yaml

# 2. Label your GPU node
kubectl label node <your-node-name> nvidia.com/gpu.present=true

# 3. Taint GPU nodes so only GPU workloads land on them
kubectl taint node <your-node-name> nvidia.com/gpu=present:NoSchedule

# 4. Deploy
kubectl apply -k k8s/

# 5. Check status
kubectl -n mnemonicai get pods

# 6. View logs
kubectl -n mnemonicai logs -f deployment/mnemonicai

# 7. Port-forward for local access
kubectl -n mnemonicai port-forward svc/mnemonicai 8400:8400 8401:8401
```

### Trigger a training run on Kubernetes

```bash
kubectl -n mnemonicai apply -f k8s/train-job.yaml
kubectl -n mnemonicai logs -f job/mnemonicai-train-run
```

### What the K8s manifests include

| File | What it does |
|------|-------------|
| `k8s/namespace.yaml` | Creates a `mnemonicai` namespace (isolates everything) |
| `k8s/configmap.yaml` | Stores `config.json` as a Kubernetes ConfigMap |
| `k8s/pvc.yaml` | Creates persistent volumes for data and models |
| `k8s/deployment.yaml` | Runs the inference server on 1 GPU |
| `k8s/service.yaml` | Exposes the API internally and externally |
| `k8s/servicemonitor.yaml` | Tells Prometheus to scrape metrics |
| `k8s/train-job.yaml` | Template for running QLoRA training as a Job |
| `k8s/kustomization.yaml` | Ties everything together for `kubectl apply -k` |

### Prometheus integration

If you're using `kube-prometheus-stack`, the `ServiceMonitor` in `k8s/servicemonitor.yaml` automatically configures Prometheus to scrape your MnemonicAi pods. The label `release: prometheus` in the ServiceMonitor must match your Prometheus installation's release name.

---

## 14. Configuration Reference

All settings live in `config.json`. After running `install.py`, this file is created automatically. You can edit it at any time — the server picks up changes on restart.

### Model settings

| Key | Default | What it does |
|-----|---------|-------------|
| `model_path` | `models/ornith-1.0-9b` | Where the raw HuggingFace model files are |
| `gguf_path` | `models/ornith-1.0-9bgguf/...Q4_K_M.gguf` | Where the compressed GGUF model is |
| `adapter_dir` | `mnemonicai_data/adapter` | Where adapter files are saved |
| `data_dir` | `mnemonicai_data` | Where the memory database and logs live |

### Training settings (QLoRA)

| Key | Default | What it does | When to change |
|-----|---------|-------------|----------------|
| `train_lr` | `2e-4` | Learning rate — how fast the AI learns | Lower to `1e-4` if training is unstable |
| `train_steps` | `8` | Number of training steps per bake | Increase to `20` if loss isn't decreasing |
| `lora_r` | `16` | LoRA rank — how much the adapter can learn | Higher = more capacity but more memory |
| `lora_alpha` | `64` | LoRA scaling factor (alpha/r = 4×) | Keep at 4× the rank |
| `lora_dropout` | `0.05` | Regularization to prevent overfitting | Increase to `0.1` if overfitting |
| `qlora_batch_size` | `4` | Examples processed per step | **Lower to `2` if you get OOM** |
| `qlora_grad_accum` | `4` | Steps to accumulate before updating | Effective batch = batch_size × grad_accum |
| `qlora_gradient_checkpointing` | `true` | Trades speed for memory savings | Keep `true` — required for batch 4 |
| `qlora_max_seq_length` | `1024` | Maximum tokens per example | Lower to `512` to save memory |
| `qlora_packing` | `true` | Concatenates examples to eliminate padding | Keep `true` — 2-3× speedup |
| `qlora_max_grad_norm` | `0.3` | Gradient clipping — prevents runaway updates | Keep at `0.3` (QLoRA paper value) |
| `qlora_warmup_ratio` | `0.03` | Fraction of steps used for warmup | Keep at `0.03` |
| `qlora_lr_scheduler` | `cosine` | How learning rate changes over time | `cosine` is best for QLoRA |
| `qlora_weight_decay` | `0.0` | Regularization on weights | Keep `0.0` — don't decay LoRA weights |
| `qlora_dataloader_workers` | `2` | CPU threads for loading data | Increase to `4` if GPU is starved |
| `qlora_seed` | `42` | Random seed for reproducibility | Same seed + same data = same adapter |
| `qlora_optim` | `paged_adamw_8bit` | Optimizer (saves memory) | Keep as-is — 4× less optimizer VRAM |

### Tuning guide

| Problem | Solution |
|---------|---------|
| OOM (out of memory) during training | Set `qlora_batch_size` to `2` or `1` |
| Loss not decreasing | Increase `train_steps` from `8` to `20` |
| Training is unstable (loss spikes) | Lower `train_lr` from `2e-4` to `1e-4` |
| Training is too slow | Set `qlora_packing` to `true` (should already be) |
| Adapter doesn't learn enough | Increase `lora_r` from `16` to `32` |
| Overfitting (good on training, bad on eval) | Increase `lora_dropout` from `0.05` to `0.1` |

---

## 15. Troubleshooting

### "CUDA not available" or "no NVIDIA GPUs detected"

**Cause:** The NVIDIA driver isn't installed or isn't loaded.

**Fix:**

```bash
# Check if the driver is loaded
nvidia-smi

# If it says "command not found" or shows no GPUs:
sudo apt update
sudo apt install -y nvidia-driver-560
sudo reboot

# After reboot, check again:
nvidia-smi
```

Then re-run the installer:

```bash
cd ~/MnemonicAi
./mn_install.sh
```

### "OOM" (Out of Memory) during training

**Cause:** The RTX 3090 ran out of VRAM (24 GB).

**Fix:** Edit `config.json`:

```json
{
  "qlora_batch_size": 2
}
```

If still OOM:

```json
{
  "qlora_batch_size": 1,
  "qlora_max_seq_length": 512
}
```

### "venv not found" when running mn_run.sh

**Cause:** The install script didn't complete successfully.

**Fix:** Re-run the installer:

```bash
cd ~/MnemonicAi
./mn_install.sh
```

Check the summary at the end for any failures. Fix them and re-run.

### The server won't start

**Debug steps:**

```bash
# 1. Check if something is already using port 8400
sudo lsof -i :8400

# 2. Try running in foreground to see the error
./mn_run.sh serve

# 3. Check the logs
./mn_run.sh logs 500

# 4. Verify the model files exist
ls -la models/ornith-1.0-9b/
ls -la models/ornith-1.0-9bgguf/
```

### Docker can't see the GPUs

**Cause:** NVIDIA Container Toolkit isn't installed or configured.

**Fix:**

```bash
# Install the toolkit
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

If you see GPU output, Docker can access your GPUs.

### "permission denied" when running scripts

**Fix:**

```bash
chmod +x mn_*.sh
```

### PCIe x4 warning during install

**Cause:** One of your GPUs is in a PCIe x4 slot, which is slower than x8 or x16. This will slow down training significantly.

**Fix:** Move the affected GPU to a slot that supports x8 or x16. Check your motherboard manual to see which slots support full bandwidth.

### Grafana shows no data

**Cause:** Prometheus can't reach the MnemonicAi container.

**Fix:**

```bash
# Check if Prometheus can see the target
curl http://localhost:9090/api/v1/targets | python3 -m json.tool | grep mnemonicai

# Check if MnemonicAi is exposing metrics
curl http://localhost:8400/metrics | head -5
```

If metrics are empty, make sure `app/metrics.py` is imported in `start.py`:

```python
from app.metrics import setup_metrics
setup_metrics(app)
```

### Grafana login fails with Google OAuth

**Common causes:**

1. The redirect URI in Google Cloud Console doesn't match exactly. It must be `http://localhost:3000/login/google` (or your domain).
2. The `.env` file is not in the same directory as `docker-compose.yml`.
3. The environment variables in `.env` have typos.
4. The `hosted_domain` restriction is blocking your email. Remove `GF_AUTH_GOOGLE_HOSTED_DOMAIN` to allow any Google account.

### Training loss isn't decreasing

**Possible causes and fixes:**

1. **Learning rate too high** — lower `train_lr` from `2e-4` to `1e-4`
2. **Not enough steps** — increase `train_steps` from `8` to `20`
3. **Batch size too small** — increase `qlora_batch_size` (if you have VRAM)
4. **Data quality** — check that the training examples are well-formatted
5. **Gradient clipping too aggressive** — check `qlora_max_grad_norm` is `0.3` (not lower)

### The adapter makes things worse

**Fix:** Roll back immediately:

```bash
./mn_adapter.sh rollback
```

Then investigate why the training failed:

```bash
./mn_adapter.sh compare v2 v3
```

Check if the evaluation loss went up — if so, the adapter overfit or the training data was noisy.

---

## 16. Script Reference

| Script | Command | What it does |
|--------|---------|-------------|
| `mn_install.sh` | `./mn_install.sh` | One-time setup: OS, Python, CUDA, PyTorch, QLoRA stack, GPU enumeration |
| `mn_run.sh` | `./mn_run.sh serve` | Start inference server in foreground |
| `mn_run.sh` | `./mn_run.sh background` | Start inference server in background |
| `mn_run.sh` | `./mn_run.sh stop` | Stop the background server |
| `mn_run.sh` | `./mn_run.sh status` | Show server status, GPU usage, port info |
| `mn_run.sh` | `./mn_run.sh logs [N]` | Show last N log lines (default 100) |
| `mn_run.sh` | `./mn_run.sh train` | Run a QLoRA training bake on GPU 1 |
| `mn_run.sh` | `./mn_run.sh mock` | Start with mock backend (no GPU) |
| `mn_run.sh` | `./mn_run.sh env` | Show environment and GPU configuration |
| `mn_run.sh` | `./mn_run.sh adapter-ui` | Start the adapter management web UI |
| `mn_backup.sh` | `./mn_backup.sh` | Create a compressed backup with rotation |
| `mn_service.sh` | `sudo ./mn_service.sh` | Install systemd service + health timer |
| `mn_diff.sh` | `MN_FLEET="user@host1,user@host2" ./mn_diff.sh` | Compare stack versions across fleet |
| `mn_adapter.sh` | `./mn_adapter.sh list` | List all adapter versions |
| `mn_adapter.sh` | `./mn_adapter.sh activate v3` | Activate a specific adapter version |
| `mn_adapter.sh` | `./mn_adapter.sh rollback` | Roll back to previous adapter |
| `mn_adapter.sh` | `./mn_adapter.sh compare v2 v3` | Compare two adapter versions |
| `mn_adapter.sh` | `./mn_adapter.sh gc 5` | Delete old versions, keep 5 most recent |

### Environment variables you can set

| Variable | Default | What it does |
|----------|---------|-------------|
| `MN_REPO` | Current directory | Path to the MnemonicAi repository |
| `MN_PORT` | `8400` | Port for the inference API |
| `MN_HOST` | `0.0.0.0` | Network interface to bind |
| `MN_INFER_GPU` | `0` | Which GPU to use for inference |
| `MN_TRAIN_GPU` | `1` | Which GPU to use for training |
| `MN_SKIP_REBOOT` | `0` | Set to `1` to skip auto-reboot during install |
| `MN_ALERT_EMAIL` | (empty) | Email address for crash alerts |
| `MN_ALERT_WEBHOOK` | (empty) | Webhook URL for crash alerts (Slack, Discord) |
| `MN_BACKUP_OFFSITE` | `/mnt/nas/mnemonicai_backups` | Offsite backup destination |
| `MN_FLEET` | (empty) | Comma-separated list of `user@host` for fleet diff |
| `GRAFANA_PASSWORD` | `admin` | Grafana admin password |
| `GRAFANA_GOOGLE_CLIENT_ID` | (empty) | Google OAuth client ID |
| `GRAFANA_GOOGLE_CLIENT_SECRET` | (empty) | Google OAuth client secret |
| `GRAFANA_HOSTED_DOMAIN` | `yourdomain.com` | Restrict Google login to this domain |

---

## Important Notes

1. **First install will reboot** after installing the NVIDIA driver (unless you set `MN_SKIP_REBOOT=1`). After the reboot, run `./mn_install.sh` again to continue.

2. **Never use the `latest` Docker tag in production.** Always pin to a specific version tag or commit SHA so you know exactly what's running and can roll back.

3. **Adapters are not auto-versioned by training code yet.** Make sure your `SleepTrainer` or `HybridBackend` calls `register_adapter()` after each successful bake (see [Section 7](#7-training-the-model-qlora-bakes)).

4. **PCIe x4 slots throttle training.** The installer warns if a GPU is in a slow slot. Move it to an x8 or x16 slot for full bandwidth.

5. **The health timer does not auto-restart the server.** This is intentional — if the server crashes, you want to investigate, not silently restart. Configure alerts (email or webhook) so you're notified when it goes down.

6. **Backups include the SQLite database, adapters, and all data.** They do not include the model files (those are large and rarely change). Make sure model files are backed up separately.

---

## File Structure

```
MnemonicAi/
├── mn_install.sh              # Installer
├── mn_run.sh                  # Runtime dispatcher
├── mn_backup.sh               # Backup script
├── mn_service.sh              # systemd installer
├── mn_diff.sh                 # Fleet diff tool
├── mn_adapter.sh              # Adapter version manager
├── mn_lib.sh                  # Shared config (sourced by all scripts)
├── config.json                # Main configuration
├── requirements.txt           # Python dependencies (base)
├── requirements-gpu.txt      # Python dependencies (GPU stack)
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose stack
├── docker-compose.observability.yml  # Prometheus + Grafana stack
├── .dockerignore
├── .env                       # Secrets (DO NOT COMMIT)
├── .gitignore
├── start.py                   # Main application entry point
├── install.py                 # Config file generator
├── dryrun_train.py            # Training script
├── train_check.py             # Training stack validation
├── app/
│   ├── adapter_ui.py          # Adapter management web UI
│   └── metrics.py             # Prometheus instrumentation
├── k8s/                       # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── servicemonitor.yaml
│   ├── train-job.yaml
│   └── kustomization.yaml
├── .github/workflows/         # CI/CD pipelines
│   ├── docker-build-push.yaml
│   └── release.yaml
├── prometheus/                # Prometheus config
│   ├── prometheus.yml
│   └── alerts.yml
├── grafana/                   # Grafana config
│   ├── dashboard.json
│   ├── datasource.yaml
│   └── dashboards.yaml
├── models/                    # Model files (not in git)
│   ├── ornith-1.0-9b/
│   └── ornith-1.0-9bgguf/
└── mnemonicai_data/           # Runtime data (not in git)
    ├── memory.db              # SQLite memory database
    ├── adapter/               # Trained adapters
    │   ├── v1/
    │   ├── v2/
    │   └── .registry.json     # Adapter version registry
    ├── logs/
    └── backups/
```