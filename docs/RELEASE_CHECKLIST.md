# Public Release Checklist

## Repository content

- [ ] Run `stage_github_release.ps1` to create a clean staging directory.
- [ ] Run `audit_github_release.ps1` and manually inspect every flagged line.
- [ ] Confirm that no raw BDD100K images, source labels, private files, model weights, API keys, or local caches are included.
- [ ] Confirm that `data/road200_manifest.csv` and `data/road200_category_mapping.csv` are present.
- [ ] Confirm that all reported CSV values match the final paper.

## Script portability

- [ ] Replace all local absolute paths and user-specific paths with CLI arguments or a configuration file.
- [ ] Document required input folders and output folders.
- [ ] Export the final environment from the actual evaluation environment.
- [ ] Test each published script from a clean clone on a different directory.
- [ ] Record the exact package versions used for the reported results.

## Dataset and publication policy

- [ ] Verify the applicable BDD100K data-use and redistribution terms.
- [ ] Do not upload raw data or derivative artifacts until their redistribution status is confirmed.
- [ ] Select a repository license after all authors agree.
- [ ] Add the final GitHub URL to the paper's Data and Code Availability statement.
- [ ] Add a repository citation after the paper is submitted or accepted.

## Final Git commands

```powershell
cd <release-directory>
git init
git add .
git status
git commit -m "Release Road200 evaluation code and reported results"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

