#!/bin/bash

# Prepare release script
# Usage: ./scripts/prepare_release.sh <version>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.2.3"
    exit 1
fi

VERSION=$1

# Update version in Cargo.toml
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" backend/Cargo.toml

# Update version in pyproject.toml
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" backend/crates/shnn-python/pyproject.toml

# Update CHANGELOG.md (append new version entry)
echo "# $VERSION" >> CHANGELOG.md
echo "" >> CHANGELOG.md
echo "## Changes" >> CHANGELOG.md
echo "- " >> CHANGELOG.md
echo "" >> CHANGELOG.md

echo "Version updated to $VERSION"
echo "Please edit CHANGELOG.md with the changes"
echo "Then run: git add . && git commit -m \"Release $VERSION\" && git tag v$VERSION"