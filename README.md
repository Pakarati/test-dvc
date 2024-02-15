# DVC Guide

## Initial Set-up

1. Install DVC in your environment and all the dependencies you need [s3], [gdrive], [azure], [ssh], etc. To install all use [all]:
```
pip install dvc[gdrive]
```

2. Initialize DVC repository:
```
$ dvc init
```

3. Create a folder in your ``Gdrive`` unit and get the `id` from url:
```
https://drive.google.com/drive/folders/<id>
```

Then, set that folder as your remote storage for DVC:

```
$ dvc remote add -d storage gdrive://<id>
```

The first time you connect/access to gdrive, DVC will ask you to authenticate. Make sure to have the corresponding permissions from your storage

4. Push the config files to your repository:
```
$ git commit .dvc/config -m <message>
$ git push
```


## Adding data files to storage

1. This adds the file to your DVC ``.gitignore`` and creates a `.dvc` file that keeps track of the file:
```
$ dvc add <data_folder/file>
```

2. Add the newly created files to your repository:
```
$ git add <data_folder>/.gitignore <data_folder/file>.dvc
$ git commit -m <message>
$ git push
```

3. Make sure to always push your changes to your remote storage:
```
$ dvc push
```


## Making changes in existing files

1. Make the changes in your file and add to DVC track:
```
$ dvc add <data_folder/file>
```

2. Add the changed `.dvc` file to your repository to keep track of the new version:
```
$ git add <data_folder/file>.dvc
$ git commit -m <message>
$ git push
```

3. Make sure to push the new change to the remote storage:
```
$ dvc push
```


## Pull data from storage

1. Like the `git status`, make sure to check the status of the files in the remote storage:
```
$ dvc data status 
or
$ dvc status
```

2. Fetch all the untracked files to your cache:
```
$ dvc fetch <untracked_files>
```

3. Finally, pull from the remote storage:
```
$ dvc pull
```


## Returning to previous version of data

1. Checkout to the previous commit in git and DVC:
```
$ git checkout HEAD^1 <data_folder/file>
$ dvc checkout
```
DVC checkout modifies the <data_folder/file> to the previous version

2. If you want to keep that version in track, commit the changes:
```
$ git commit <data_folder/file>.dvc -m <message>
$ git push
```

3. Add the new track file to dvc:
```
$ dvc add
```

4. No need to push, since we already have that version in our storage


## Switching beetween versions
1. Checkout to the desired commit or branch in git, and then checkout in DVC:
```
$ git checkout <...> <...>.dvc
$ dvc checkout
```

2. Follow the steps above


## Get storaged file from DVC Roposiroty with Python API

1. Install DVC in your environment:
```
pip install dvc
```

2. Import DVC API in your code:
```
import dvc.api
```

3. To get a file from DVC repository:
```
with dvc.api.open("path_to_file", repo=<repo_url>) as fd:
    #stream files
    fd.read()
```

## Other commands:

* `$ dvc list <address_dvc_repository>`: List all files in DVC repository
* `$dvc diff <a_rev> <b_rev>`: Compares 2 commits and shows added, modified an deleted DVC tracked files.

    <a_rev>: Old commit. Default HEAD

    <b_rev>: New commit. Default to current workspace

* `dvc doctor`: Display DVC version, environment and project information