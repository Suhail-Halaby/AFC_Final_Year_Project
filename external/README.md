# External code

Third-party repositories pinned as git submodules, so experiments are tied to an exact upstream commit.
For example:

```bash
git submodule add https://github.com/dynamicslab/hydrogym external/hydrogym
```

Do not edit code in here directly — subclass or wrap it from `src/afc/` or `environments/hydrogym/`.
