import boto.emr
conn = boto.emr.connect_to_region(
            'us-west-1',
            aws_access_key_id='AKIAJ3WMYTNKB77YZ5KQ',
            aws_secret_access_key='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')
conn.terminate_jobflow('j-3RUN8W208F7S3')