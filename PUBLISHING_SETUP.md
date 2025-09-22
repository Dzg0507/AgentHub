# 🚀 Publishing Setup Guide

This guide will help you set up GitHub Actions for automated publishing to PyPI and creating GitHub releases.

## 🔑 Required Secrets

### 1. PYPI_API_TOKEN

This token allows GitHub Actions to publish your package to PyPI.

#### **Step 1: Create PyPI Account**
1. Go to [pypi.org](https://pypi.org)
2. Click **"Register"** to create an account
3. Verify your email address

#### **Step 2: Create API Token**
1. Log in to PyPI
2. Go to **Account Settings** → **API tokens**
3. Click **"Add API token"**
4. Fill in the form:
   - **Token name**: `Agent Control Hub Release`
   - **Scope**: `Entire account (all projects)`
5. Click **"Add token"**
6. **Copy the token** (starts with `pypi-`) - you won't see it again!

#### **Step 3: Add to GitHub Secrets**
1. Go to your GitHub repository: https://github.com/Dzg0507/AgentHub
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **"New repository secret"**
5. Fill in:
   - **Name**: `PYPI_API_TOKEN`
   - **Secret**: Paste your PyPI token
6. Click **"Add secret"**

## 🏷️ Creating Releases

### Method 1: Using GitHub Web Interface

1. Go to your repository on GitHub
2. Click **"Releases"** tab
3. Click **"Create a new release"**
4. Fill in:
   - **Tag version**: `v1.0.0` (or your version)
   - **Release title**: `Release v1.0.0`
   - **Description**: Copy from CHANGELOG.md
5. Click **"Publish release"**

### Method 2: Using Git Commands

```bash
# Create and push a tag
git tag v1.0.0
git push origin v1.0.0

# This will automatically trigger the release workflow
```

### Method 3: Using GitHub CLI

```bash
# Install GitHub CLI first: https://cli.github.com/
gh release create v1.0.0 --title "Release v1.0.0" --notes "Initial release"
```

## 🔄 Workflow Triggers

The release workflow will automatically run when:

- **Push a tag** starting with `v` (e.g., `v1.0.0`)
- **Manual trigger** via GitHub Actions tab

## 📦 What Happens During Release

### 1. **Testing Phase**
- Runs all tests to ensure code quality
- Checks code coverage
- Verifies imports work correctly

### 2. **Building Phase**
- Creates source distribution (`.tar.gz`)
- Creates wheel distribution (`.whl`)
- Validates package structure

### 3. **Publishing Phase**
- Uploads package to PyPI (if `PYPI_API_TOKEN` is set)
- Creates GitHub release with assets
- Notifies about success/failure

## 🧪 Testing the Workflow

### Test Without Publishing

1. **Create a test tag**:
   ```bash
   git tag v0.0.1-test
   git push origin v0.0.1-test
   ```

2. **Check GitHub Actions**:
   - Go to **Actions** tab in your repository
   - Look for the "Release" workflow
   - It should run but skip PyPI publishing (no token)

3. **Delete test tag**:
   ```bash
   git tag -d v0.0.1-test
   git push origin --delete v0.0.1-test
   ```

### Test With Publishing

1. **Set up PyPI token** (see steps above)
2. **Create a real release**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
3. **Monitor the workflow** in GitHub Actions
4. **Check PyPI** for your package: https://pypi.org/project/agent-control-hub/

## 🔧 Troubleshooting

### Common Issues

#### 1. **"PYPI_API_TOKEN not found"**
- **Solution**: Make sure you added the secret correctly
- **Check**: Go to Settings → Secrets and variables → Actions

#### 2. **"Package already exists"**
- **Solution**: Increment version number in `setup.py`
- **Check**: PyPI doesn't allow overwriting existing versions

#### 3. **"Invalid credentials"**
- **Solution**: Regenerate PyPI token and update GitHub secret
- **Check**: Token might have expired or been revoked

#### 4. **"Workflow not triggered"**
- **Solution**: Make sure tag starts with `v` (e.g., `v1.0.0`)
- **Check**: Tag must be pushed to the main branch

### Debugging Steps

1. **Check workflow logs**:
   - Go to Actions tab
   - Click on the failed workflow
   - Look at the logs for error details

2. **Verify secrets**:
   - Go to Settings → Secrets and variables → Actions
   - Make sure `PYPI_API_TOKEN` exists and is correct

3. **Test locally**:
   ```bash
   # Test package building
   python -m build
   
   # Test package validation
   twine check dist/*
   ```

## 📋 Pre-Release Checklist

Before creating a release, make sure:

- [ ] **Version updated** in `setup.py`
- [ ] **CHANGELOG.md updated** with new features/fixes
- [ ] **All tests passing** locally
- [ ] **README.md updated** if needed
- [ ] **PyPI token** added to GitHub secrets
- [ ] **Code reviewed** and approved
- [ ] **Documentation updated**

## 🎯 Release Process

### 1. **Prepare Release**
```bash
# Update version in setup.py
# Update CHANGELOG.md
# Commit changes
git add .
git commit -m "chore: prepare release v1.0.0"
git push origin main
```

### 2. **Create Release**
```bash
# Create and push tag
git tag v1.0.0
git push origin v1.0.0
```

### 3. **Monitor Progress**
- Watch GitHub Actions tab
- Check PyPI for package upload
- Verify GitHub release creation

### 4. **Verify Release**
- **PyPI**: https://pypi.org/project/agent-control-hub/
- **GitHub**: https://github.com/Dzg0507/AgentHub/releases
- **Installation**: `pip install agent-control-hub`

## 🎉 Success!

Once everything is set up, you can:

- **Automatically publish** to PyPI with every release
- **Create GitHub releases** with changelog and assets
- **Distribute your package** easily to users
- **Track releases** with proper versioning

Your Agent Control Hub will be available for installation via:
```bash
pip install agent-control-hub
```

## 📞 Getting Help

If you run into issues:

1. **Check the logs** in GitHub Actions
2. **Verify secrets** are set correctly
3. **Test locally** before creating releases
4. **Open an issue** on GitHub if problems persist

Happy publishing! 🚀
