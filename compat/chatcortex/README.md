# chatcortex is now FrontierKit

This library was renamed. The code, the documentation and all new releases are
under the new name:

- PyPI: [frontierkit](https://pypi.org/project/frontierkit/)
- Website: [frontierkit.github.io/frontierkit](https://frontierkit.github.io/frontierkit/)
- Source: [github.com/frontierkit/frontierkit](https://github.com/frontierkit/frontierkit)

## What to change

```bash
pip uninstall chatcortex
pip install frontierkit
```

Then replace `chatcortex` with `frontierkit` in your imports. Nothing else
about the API changed. The commands are now `frontierkit` and
`frontierkit-registry`.

## What this package does

`chatcortex` 0.13.0 has no code of its own. It installs `frontierkit`, makes
`import chatcortex` and `from chatcortex.anything import ...` load the same
modules from `frontierkit`, keeps the `chatcortex` and `chatcortex-registry`
commands, and warns once that the name has changed. It will get no further
releases, so new features arrive only in `frontierkit`.
