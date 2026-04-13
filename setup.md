# Lab Server User Guide

> **Your Login:** `your_username@10.225.67.239`  
> **Example:** If your username is `eet242799`, then use `ssh eet242799@10.225.67.239`

---

## Terminology

| Term | Meaning |
|------|---------|
| **Personal Machine** | Your laptop/desktop (where you open Terminal) |
| **Lab Server** | The IITD lab machine (accessed via SSH) |

---

## 1. First-Time Setup (One-Time Only)

> Do the following steps **on the Lab Server** after your first login.

### 1.1 Login to the Server

From your **Personal Machine**, open a terminal and run:

```bash
ssh eet242799@10.225.67.239
```

Replace `eet242799` with your username.  
Enter your password when prompted.

### 1.2 Check Required Files

After logging in, run:

```bash
ls
```

Ensure these files exist:
- `proxy.sh`
- `CCIITD-CA.crt`

> âš ï¸ If missing, contact the lab admin, visit the [CSC website](https://csc.iitd.ac.in), or visit the CSC center to get them.

### 1.3 Edit proxy.sh with Your Credentials

Open the file:

```bash
nano proxy.sh
```

Inside the file:
- Add your **IITD Kerberos username**
- Add your **IITD Kerberos password**
- Update the **proxy URL** according to your category:

```
proxy_add=https://proxy82.iitd.ac.in/cgi-bin/proxy.cgi
```

Change the proxy number based on your program. For PhD use 61, for MTech/MS use 62, for BTech use 22, for Staff use 82. For other categories, check the CSC website.

Save and exit:
- `Ctrl + O` â†’ `Enter` (save)
- `Ctrl + X` (exit)

### 1.4 Make Script Executable

```bash
chmod +x proxy.sh
```

Initial setup is now complete.

### 1.5 (Recommended) SSH Key Setup

Password login works, but SSH key-based login is recommended for convenience and security. It will save you from entering your password every time you connect.

#### Linux / macOS

**Step 1:** Generate SSH key (on your Personal Machine):
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Press Enter to accept default location. Set a passphrase if desired.

**Step 2:** Copy key to server:
```bash
ssh-copy-id eet242799@10.225.67.239
```

#### Windows (PowerShell)

**Step 1:** Generate SSH key:
```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
```

**Step 2:** Copy key to server:
```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh eet242799@10.225.67.239 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

#### If `ssh-copy-id` Doesn't Work

Manually copy your public key:

**Step 1:** View your public key (Personal Machine):
```bash
cat ~/.ssh/id_ed25519.pub
```

**Step 2:** Login to server and add the key:
```bash
ssh eet242799@10.225.67.239
mkdir -p ~/.ssh
nano ~/.ssh/authorized_keys
```
Paste your public key, save and exit.

**Step 3:** Set correct permissions:
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

---

## 2. Daily Login Workflow

### 2.1 SSH from Personal Machine

```bash
ssh eet242799@10.225.67.239
```

Replace `eet242799` with your username.

### 2.2 Activate Internet (Proxy Login)

On the **Lab Server**, run:

```bash
./proxy.sh &
```

You should see:
```
proxy login
```
or
```
Already logged in
```

### 2.3 Set Proxy Environment Variables

Run one of the following based on your program:

**PhD Students (port 61):**
```bash
export http_proxy="http://proxy61.iitd.ac.in:3128"
export https_proxy="http://proxy61.iitd.ac.in:3128"
```

**MTech/MS Students (port 62):**
```bash
export http_proxy="http://proxy62.iitd.ac.in:3128"
export https_proxy="http://proxy62.iitd.ac.in:3128"
```

> For other categories, check the port number on the [CSC website](https://csc.iitd.ac.in).

### 2.4 Verify Internet Access

```bash
wget www.google.com
```

If `index.html` is created, internet is working!

Optional cleanup:
```bash
rm index.html
```

---

## 3. Using Python/Conda

### 3.1 Activate Shared Environment

The server has a pre-installed environment called `aiml`:

```bash
conda activate aiml
```

Your prompt will change to:
```
(aiml) eet242799@10.225.67.239:~$
```

**Included packages:**
- Python 3.10
- NumPy, SciPy, Pandas
- Scikit-learn
- PyTorch (CPU)
- OpenCV
- Jupyter Notebook/Lab

### 3.2 âš ï¸ Important Rule

> **DO NOT install packages directly into `aiml`.**  
> This environment is shared by all users!

### 3.3 Create Your Own Environment (Recommended)

If you need extra packages:

```bash
conda create -n eet242799ENV --clone aiml
conda activate eet242799ENV
```

Now you can safely install:
```bash
pip install <package-name>
```

This will not affect other users.

---

## 4. Using JupyterLab Remotely

Jupyter runs on the **Lab Server** but is accessed from your **Personal Machine** browser.

### 4.1 Start JupyterLab (on Lab Server)

```bash
conda activate aiml
jupyter lab --no-browser --port=8889
```

You will see output like:
```
http://localhost:8889/?token=XXXXXXXX
```

> âš ï¸ **Do NOT close this terminal!**

### 4.2 Create SSH Tunnel (on Personal Machine)

Open a **NEW terminal** on your **Personal Machine** and run:

```bash
ssh -N -L 8890:localhost:8889 eet242799@10.225.67.239
```

Replace `eet242799` with your username.

> This terminal will appear idle â€” this is expected.

### 4.3 Open Jupyter in Browser

On your **Personal Machine**, open your browser and go to:

```
http://localhost:8890
```

Paste the **token** from Step 4.1 when prompted.

### 4.4 Select Correct Kernel

Inside Jupyter:
1. Go to **Kernel** â†’ **Change Kernel**
2. Select: `Python 3.10 (AIML)` or your personal environment

---

## 5. Logging Out

When finished, simply run:

```bash
exit
```

---

## 6. Troubleshooting

| Issue | Solution |
|-------|----------|
| Can't connect to server | Check your internet connection, verify IP and username are correct. Server is only accessible from IIT intranet â€” use VPN if you are off-campus. |
| Internet not working | Check proxy.sh has correct username, password, and proxy port. Verify `CCIITD-CA.crt` exists. Ensure you exported proxy with the correct port number. |
| `conda activate` doesn't work | Run `conda init bash` then restart terminal |
| Jupyter token not working | SSH tunnel is required! Without proper SSH tunnel setup, it will not work. Ensure Step 4.2 is completed. |
| Permission denied on proxy.sh | Run `chmod +x proxy.sh` |