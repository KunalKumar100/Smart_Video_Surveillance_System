#!/bin/bash

# Ask for GitHub repo URL
read -p "https://github.com/KunalKumar100/Video_Surveillance.git:" repo_url

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing Git..."
    git init
else
    echo "Git already initialized."
fi

# Add remote origin
git remote add origin "$repo_url" 2>/dev/null || git remote set-url origin "$repo_url"

# Rename branch to main (safe default)
git branch -M main

# Add and commit all files
git add .
git commit -m "Initial commit"

# Push to GitHub
git push -u origin main

echo "✅ Project connected and pushed to GitHub!"
