# DVC Guide

## Initial Set-up

1. Initialize DVC repository:
```
$ dvc init
```

2. Create a folder in your ``Gdrive`` unit and get the `id` from url:
```
https://drive.google.com/drive/folders/<id>
```

Then, set that folder as your remote storage for DVC:

```
$ dvc remote add -d storage gdrive://<id>
```

The first time you connect/access to gdrive, DVC will ask you to authenticate. Make sure to have the corresponding permissions from your storage

3. Push the config files to your repository:
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

2. If you want to keep that version in track, commit the changes:
```
$ git commit <data_folder/file>.dvc -m <message>
$ git push
```

3. No need to push to, since we already have that version in our storage


## Switching beetween versions
1. Checkout to the desired commit or branch in git, and then checkout in DVC:
```
$ git checkout <...> <...>.dvc
$ dvc checkout
```