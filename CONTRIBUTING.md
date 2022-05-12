# How to Contribute

Basic Pitch welcomes your contributions!

## Getting Started

To get your environment set up to build `basic-pitch`, you'll need Python and Pip installed on your machine.

We recommend first installing the following non-python dependencies:

- [libsndfile](http://libsndfile.github.io/libsndfile/) is a C library for reading and writing files containing sampled sound through one standard library interface.
    - To install on MacOs, run `brew install libsndfile` using [Homebrew](https://brew.sh/)
    - To install on Windows, run `choco install libsndfile` using [Chocolatey](https://chocolatey.org/)
    - To install on Ubuntu, run `sudo apt-get update && sudo apt-get install --no-install-recommends -y --fix-missing pkg-config libsndfile1`
- [ffmpeg](https://ffmpeg.org/) is a complete, cross-platform solution to record, convert and stream audio in all `basic-pitch` supported formats

To compile a debug build of `basic-pitch` that allows using a debugger (like gdb or lldb), use the following command to build the package locally and install a symbolic link for debugging:
```shell
python3 setup.py build develop
```

Then, you can `import basic_pitch` from Python (or run the tests with `tox`) to test out your local changes.

## Workflow

We follow the [GitHub Flow Workflow](https://guides.github.com/introduction/flow/):

1.  Fork the project
1.  Check out the `main` branch
1.  Create a feature branch
1.  Write code and tests for your change
1.  From your branch, make a pull request against `https://github.com/spotify/basic-pitch`
1.  Work with repo maintainers to get your change reviewed
1.  Wait for your change to be pulled into `https://github.com/spotify/basic-pitch/main`
1.  Delete your feature branch

## Testing

We use `tox` for testing - running tests from end-to-end should be as simple as:

```
tox
```

## Style

Use `black` with defaults for Python code.

## Issues

When creating an issue please try to ahere to the following format:

    module-name: One line summary of the issue (less than 72 characters)

    ### Expected behaviour

    As concisely as possible, describe the expected behaviour.

    ### Actual behaviour

    As concisely as possible, describe the observed behaviour.

    ### Steps to reproduce the behaviour

    List all relevant steps to reproduce the observed behaviour.

## Documentation

We also welcome improvements to the project documentation or to the existing
docs. Please file an [issue](https://github.com/spotify/basic-pitch/issues/new).

## First Contributions

If you are a first time contributor to `basic-pitch`, familiarize yourself with the:
* [Code of Conduct](CODE_OF_CONDUCT.md)
* [GitHub Flow Workflow](https://guides.github.com/introduction/flow/)
<!-- * Issue and pull request style guides -->

When you're ready, navigate to [issues](https://github.com/spotify/basic-pitch/issues/new). Some issues have been identified by community members as [good first issues](https://github.com/spotify/basic-pitch/labels/good%20first%20issue).

There is a lot to learn when making your first contribution. As you gain experience, you will be able to make contributions faster. You can submit an issue using the [question](https://github.com/spotify/basic-pitch/labels/question) label if you encounter challenges.

# License

By contributing your code, you agree to license your contribution under the
terms of the [LICENSE](https://github.com/spotify/basic-pitch/blob/main/LICENSE).

# Code of Conduct

Read our [Code of Conduct](CODE_OF_CONDUCT.md) for the project.