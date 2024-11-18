# 🔥 Play gound for modal: a serverless GPU service

1. Install python package

    ```shell
    pip install -e .
    ```

1. Setup modal

    Run this command, and it will show you a URL to authentication.


    ```shell
    python -m modal setup
    ```

    Follow the instruction, then you will see an response like below.


    ```shell
    Token verified successfully!
    ```

1. Run your first modal app

    ```shell
    modal run scripts/001-calc-square.py
    ```

    Once you run the command above, you will see the output like below.

    ```shell
    ✓ Initialized. View run at https://modal.com/apps/...
    ✓ Created objects.
    ├── 🔨 Created mount /home/wattai/dev/pg-modal/scripts/001-calc-square.py
    └── 🔨 Created function square.
    This code is running on a remote worker!
    the square is 1764
    Stopping app - local entrypoint completed.
    ✓ App completed. View run at https://modal.com/apps/...
    ```

    Now that, you can see the result of `the square is 1764`.\
    🎉 Happy hacking!!
