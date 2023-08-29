# CVOps in production with Databricks Asset Bundles

This is a repository that supports TechSummit Talk 2023. 
Authors:
Anastasia Prokaieva 
Jose Alfonso

## Introduction
_Databricks Asset Bundles_, also known simply as bundles, enable you to programmatically validate, deploy, and run the projects you are working on in Databricks via the [Databricks CLI](https://github.com/databricks/cli).  A bundle is a collection of one or more related files that contain:

* Any local artifacts (such as source code) to deploy to a remote Databricks workspace prior to running any related Databricks workflows.

* The declarations and settings for the Databricks jobs, Delta Live Tables pipelines, or [MLOps Stacks](https://github.com/databricks/mlops-stack) that act upon the artifacts that were deployed into the workspace.

For more information on bundles, please see the following pages in Databricks documentation:

#### Tutorials
* [Bundle development tasks](https://docs.databricks.com/dev-tools/bundles/work-tasks.html)
* [How to use Bundles with Databricks Workflows (aka Jobs)](https://docs.databricks.com/workflows/jobs/how-to/use-bundles-with-jobs.html)
* [Automate a DLT pipeline with DABs](https://docs.databricks.com/delta-live-tables/tutorial-bundles.html)
* [Run A CI/CD process with DABs and GitHub Actions](https://docs.databricks.com/dev-tools/bundles/ci-cd.html)

#### Reference 
* [bundle settings reference](https://docs.databricks.com/dev-tools/bundles/settings.html)
* [bundle command group reference](https://docs.databricks.com/dev-tools/cli/bundle-commands.html)

## Deploying your CV model
In this repo you'll find a simple project consisting of:

1. XXX
2. XXX
3. XXX

These data assets are represented in the `bundle.yml` file in the project root directory.  

#### Deploying and running this repo
Make sure you have the Databricks CLI installed, then you can use the `databricks bundle` commands.  You'll also want to edit the `bundle.yml` and specify the Databricks Workspace you plan to deploy to.  Once you've got that sorted out, you can deploy and run the project using the following commands:

```
databricks bundle deploy
databricks bundle run XXXX
```

## Questions?
Please raise an issue if you encounter any issues and create a pull-request in case of improvements. 


### Resources 


_Click [here](https://www.youtube.com/watch?v=9HOgYVo-WTM) to watch the talk on Databricks Asset Bundles at Data & AI Summit 2023._

