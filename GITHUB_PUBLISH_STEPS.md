# GitHub Publish Steps

Recommended repository name:

- `rice-blb-multimodal-phenotyping`

Recommended visibility:

- Public, if this is the manuscript companion repository
- Private, if you want to review it before release

## 1. Create an empty GitHub repository

Create a new empty repository on GitHub named `rice-blb-multimodal-phenotyping`.
Do not add a README, .gitignore, or license on GitHub, because they already exist locally.

## 2. Open Terminal in this folder

Repository path:

- `/Users/sb2639/Desktop/rice-blb-multimodal-phenotyping`

## 3. Initialize git and make the first commit

Run:

```bash
cd /Users/sb2639/Desktop/rice-blb-multimodal-phenotyping
git init -b main
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL"
git add .
git commit -m "Initial manuscript workflow bundle"
```

## 4. Connect the GitHub remote and push

Replace `YOUR_GITHUB_USERNAME` with your GitHub username or organization name.

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/rice-blb-multimodal-phenotyping.git
git push -u origin main
```

## Notes

- The shared public inputs are `input_data/training_90.csv`, `input_data/holdout_test_10.csv`, and `input_data/training_90_grouped5fold_unit_assignments.csv`.
- These files contain the grouped 90/10 split used for the manuscript, with `cv_fold` labels attached only to the 90% training set.
- Model outputs are written to `outputs/` by default, which is ignored by git.
