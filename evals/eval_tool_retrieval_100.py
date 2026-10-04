"""Benchmark: Evaluating klix as a 100-tool Dynamic Retrieval Engine.

Simulates 100 realistic MCP tools across 10 domains:
GitHub, SQL, Kubernetes, AWS, Slack, Jira, Docker, FileSystem, Observability, Billing/CRM.

Measures:
- Top-1, Top-3, Top-5 Recall/Accuracy (Can the LLM see the correct tool?)
- Top-1 and Top-5 Latency on CPU (Fast-path vs Dense Hybrid)
- Compile time & memory footprint
"""

import time
import numpy as np
from klix import Choice, DecisionEngine

# 100 Tools: (tool_name, [anchors], [unseen_test_queries])
TOOLS_DATA = [
    # --- GitHub / Git (10 tools) ---
    ("gh_create_pull_request", 
     ["create a new pull request on github", "open a PR for code review", "submit branch changes for review"],
     ["I want to open a PR comparing main and feature-login", "submit my branch so the team can review it", "erstelle einen Pull Request für meinen Branch"]),
    ("gh_list_issues",
     ["list open issues in the repository", "search github issues with filters", "get bug reports and tickets on github"],
     ["show me all open bugs labeled critical", "which issues are currently assigned to me?", "zeige mir die offenen GitHub Issues"]),
    ("gh_merge_pull_request",
     ["merge an approved pull request", "squash and merge PR into main branch", "complete pull request review and merge"],
     ["merge PR #42 into main", "squash merge this pull request", "kannst du den PR 102 mergen?"]),
    ("git_clone_repo",
     ["clone a remote git repository locally", "download git repo via ssh or https", "git clone project from github"],
     ["clone git@github.com:foo/bar.git into ./bar", "checkout the repository to my local disk", "klone das repo auf meinen rechner"]),
    ("git_commit_changes",
     ["record changes to the repository with git commit", "commit staged files with a message", "save git working tree state"],
     ["commit these staged files with message 'fix auth bug'", "git commit -m 'update docs'", "committe die Änderungen mit einer Nachricht"]),
    ("git_interactive_rebase",
     ["rebase commits interactively git rebase -i", "squash clean up commit history before merge", "reorder or edit previous git commits"],
     ["git rebase -i HEAD~4 to squash commits", "clean up the last 3 commits into one", "interaktives rebase über die letzten commits"]),
    ("git_diff_uncommitted",
     ["show changes between commits working tree git diff", "inspect unstaged modifications in files", "view git patch of modified code"],
     ["what lines did I modify in src/?", "show git diff of working directory", "zeige mir die uncommitted differences"]),
    ("gh_create_release",
     ["create a new github release with git tag and notes", "publish a release with release notes and assets", "cut a new version release"],
     ["tag v1.2.0 and publish a new GitHub release", "create release notes for the new version", "baue ein neues Release auf GitHub"]),
    ("gh_add_review_comment",
     ["post a review comment on a pull request line", "leave inline review feedback on PR diff", "approve or request changes on github PR"],
     ["comment on line 45 that this function lacks error handling", "submit review comment on PR #12", "hinterlasse feedback zu dem code im PR"]),
    ("gh_fork_repository",
     ["fork a repository to your own github account", "create a personal fork of an upstream repo", "fork project on github"],
     ["fork this repository so I can send a PR later", "create a fork under my namespace", "forke das repo in mein profil"]),

    # --- SQL / Database (10 tools) ---
    ("sql_execute_query",
     ["execute a raw SQL SELECT query against relational database", "run SQL query and fetch result rows", "execute database SQL statement"],
     ["SELECT user_id, email FROM users WHERE active = true", "fetch the latest 10 orders from postgres", "führe die SQL-Abfrage aus"]),
    ("sql_explain_plan",
     ["explain query execution plan EXPLAIN ANALYZE", "inspect query planner cost and index scan", "analyze database query performance"],
     ["why is this slow? EXPLAIN ANALYZE SELECT * FROM orders", "inspect execution plan for full table scan", "zeige den query execution plan"]),
    ("sql_list_tables",
     ["list all tables and views in database schema", "show table names in public schema", "get database table catalog"],
     ["what tables exist in the database?", "list all relations in postgres", "zeige mir alle Tabellen in der Datenbank"]),
    ("sql_table_schema",
     ["get column definitions data types and constraints for table", "describe table structure and foreign keys", "inspect table schema columns"],
     ["what columns does the 'invoices' table have?", "describe schema of customers table", "wie ist das Schema der Tabelle 'users'?"]),
    ("sql_create_backup",
     ["create database snapshot backup pg_dump", "dump database to backup storage file", "generate full database backup archive"],
     ["take a pg_dump backup before running migration", "create a snapshot of the production db", "erstelle ein Backup der Datenbank"]),
    ("sql_restore_backup",
     ["restore database from dump file pg_restore", "recover database from backup snapshot", "apply backup file to database instance"],
     ["restore test database from yesterday's dump", "load the pg_dump file into local postgres", "spiele das Backup wieder ein"]),
    ("sql_vacuum_analyze",
     ["run VACUUM ANALYZE to reclaim storage and update planner statistics", "optimize postgres table bloat vacuum", "reindex and vacuum database"],
     ["vacuum analyze on the events table", "reclaim dead tuples in postgres", "optimiere die Tabelle mit vacuum analyze"]),
    ("sql_migrate_schema",
     ["apply database migration scripts alembic or flyway", "run pending database schema migrations", "upgrade database version schema"],
     ["apply pending alembic migrations to head", "run database schema update script", "führe die anstehenden Datenbank-Migrationen aus"]),
    ("sql_kill_blocking_query",
     ["terminate blocking database query pg_terminate_backend", "kill hanging or locked database connection", "abort long running query pid"],
     ["kill the hung query blocking transaction on pid 9281", "terminate postgres backend with pid 1234", "beende die blockierende SQL-Abfrage"]),
    ("sql_export_csv",
     ["export SQL query results to CSV file", "dump query output into downloadable csv", "save database query rows as comma separated values"],
     ["export all active subscribers to a CSV file", "save query result as customers_export.csv", "exportiere die Abfrage als CSV"]),

    # --- Kubernetes / K8s (10 tools) ---
    ("k8s_get_pods",
     ["list kubernetes pods in namespace with status", "kubectl get pods across namespaces", "check running pod status k8s"],
     ["kubectl get pods -n production", "are all backend pods running?", "zeige mir alle laufenden Pods im Cluster"]),
    ("k8s_describe_pod",
     ["describe kubernetes pod details events and containers", "kubectl describe pod for error debugging", "inspect pod failure CrashLoopBackOff"],
     ["why is pod auth-service-7f9 crash looping?", "describe pod payment-api-xyz in staging", "beschreibe den Pod und zeige Events"]),
    ("k8s_get_logs",
     ["fetch container logs from kubernetes pod", "kubectl logs -f pod container log stream", "read stdout stderr logs of k8s service"],
     ["show me logs from the worker pod for the last 10 minutes", "kubectl logs deployment/api-gateway", "hole die Logs vom Pod"]),
    ("k8s_restart_rollout",
     ["restart kubernetes rollout deployment", "kubectl rollout restart deployment", "trigger rolling restart of k8s pods"],
     ["trigger a rollout restart of deployment frontend", "restart all pods in the billing deployment", "starte das Deployment neu"]),
    ("k8s_scale_deployment",
     ["scale kubernetes deployment replica count", "kubectl scale deployment --replicas", "increase or decrease pod replicas"],
     ["scale payment-service up to 10 replicas", "scale down worker deployment to 2 instances", "skaliere das Deployment auf 5 Replicas"]),
    ("k8s_port_forward",
     ["port forward local port to kubernetes pod", "kubectl port-forward pod local connection", "tunnel network traffic to k8s service"],
     ["port-forward redis pod 6379 to localhost:6379", "open local tunnel to postgres pod on port 5432", "leite Port 8080 auf den Pod weiter"]),
    ("k8s_apply_manifest",
     ["apply kubernetes yaml manifest kubectl apply -f", "deploy k8s resources from yaml file", "apply configmap service deployment spec"],
     ["apply deployment.yaml to staging namespace", "kubectl apply -f k8s/ingress.yaml", "wende das YAML Manifest an"]),
    ("k8s_get_events",
     ["list kubernetes cluster warning events", "kubectl get events sorted by timestamp", "inspect cluster level errors and scheduling failures"],
     ["check warning events in namespace default", "why could pod not be scheduled on node?", "zeige mir die Kubernetes Events"]),
    ("k8s_cordon_node",
     ["cordon and drain kubernetes node for maintenance", "kubectl cordon node mark unschedulable", "prevent new pods from scheduling on node"],
     ["cordon worker-node-03 before kernel reboot", "mark node unschedulable", "sperre den Node für neue Pods"]),
    ("k8s_edit_configmap",
     ["update kubernetes configmap data", "kubectl edit configmap or patch config", "modify environment configuration in k8s"],
     ["update DATABASE_URL key in app-config configmap", "patch configmap with new feature flag", "ändere die ConfigMap im Cluster"]),

    # --- AWS / Cloud (10 tools) ---
    ("aws_s3_upload",
     ["upload file or object to AWS S3 bucket", "put object to s3 storage bucket", "store asset in amazon s3"],
     ["upload backup.tar.gz to s3://my-company-backups", "save this image to our assets s3 bucket", "lade die Datei in den S3 Bucket hoch"]),
    ("aws_s3_download",
     ["download object from AWS S3 bucket", "get file from s3 cloud storage", "fetch s3 object to local path"],
     ["download s3://data-lake/raw/transactions.parquet", "fetch the trained model file from s3 bucket", "lade die Datei aus dem S3 Bucket herunter"]),
    ("aws_ec2_start",
     ["start stopped EC2 virtual machine instance", "turn on amazon ec2 server instance", "aws ec2 start-instances"],
     ["boot up the dev-gpu-box ec2 instance i-0123456", "start the stopped bastion server", "starte die EC2 Instanz"]),
    ("aws_ec2_stop",
     ["stop running EC2 compute instance", "power down amazon ec2 instance", "aws ec2 stop-instances"],
     ["shut down instance i-abcdef to save cloud cost", "stop the staging server for the night", "stoppe die EC2 Instanz"]),
    ("aws_lambda_invoke",
     ["invoke AWS Lambda serverless function with payload", "trigger lambda execution and get response", "call serverless lambda function"],
     ["invoke pdf-generator-lambda with payload {'doc_id': 99}", "trigger the image-resize lambda", "führe die Lambda Funktion aus"]),
    ("aws_cloudwatch_metrics",
     ["query AWS CloudWatch metric data cpu memory latency", "get cloudwatch alarms and metric statistics", "inspect ec2 rds cloudwatch metrics"],
     ["what was the average CPU utilization on RDS for the past 2 hours?", "fetch CloudWatch metric for 5xx errors", "hole die CloudWatch Metriken"]),
    ("aws_rds_describe",
     ["describe AWS RDS database instances status and endpoint", "inspect relational database service configuration", "get rds db cluster status"],
     ["check status and replica lag of primary RDS postgres", "what is the endpoint url of the staging db?", "zeige mir den Status der RDS Instanz"]),
    ("aws_sqs_send_message",
     ["send message payload to AWS SQS queue", "enqueue task into simple queue service", "publish job message to sqs url"],
     ["send processing job to sqs://orders-queue", "publish message {'task': 'send_email'} to SQS", "sende eine Nachricht an die SQS Queue"]),
    ("aws_route53_list_records",
     ["list DNS resource record sets in AWS Route53 hosted zone", "inspect dns records A CNAME in route53", "get domain dns zone config"],
     ["what IP does api.example.com point to in Route53?", "list all CNAME records for our domain", "zeige die DNS Records in Route53"]),
    ("aws_iam_create_user",
     ["create AWS IAM user and access credentials", "provision new identity and access management user", "add cloud iam account"],
     ["create an IAM user for the new contractor with read-only policy", "provision service account in AWS IAM", "erstelle einen neuen IAM Benutzer"]),

    # --- Slack / Comms (10 tools) ---
    ("slack_send_message",
     ["send chat message to slack channel or user DM", "post notification message in slack", "post text to slack channel webhook"],
     ["send 'Deployment finished successfully' to #engineering", "notify the on-call channel in Slack", "schreibe eine Nachricht in den Slack Channel"]),
    ("slack_create_channel",
     ["create new public or private slack channel", "provision new channel in workspace", "open slack channel for project"],
     ["create private slack channel #incident-2026-10-01", "open a new channel for customer onboarding", "erstelle einen neuen Slack Kanal"]),
    ("slack_list_users",
     ["list workspace members users and emails in slack", "lookup slack user profiles in directory", "find teammate slack user id"],
     ["find the slack user ID for max@example.com", "who are all members in #dev-team?", "suche den Benutzer in Slack"]),
    ("slack_upload_file",
     ["upload file snippet or image to slack channel", "share screenshot or log file in slack conversation", "post attachment to slack"],
     ["upload error.log to the #triage channel", "share this generated chart in the slack thread", "lade die Logdatei in Slack hoch"]),
    ("slack_add_reaction",
     ["add emoji reaction to slack message", "react with emoji to chat post", "add thumbs up or checkmark emoji in slack"],
     ["add a :white_check_mark: reaction to the alert message", "react with :eyes: to show I am looking at it", "reagiere mit einem Emoji auf die Nachricht"]),
    ("slack_set_status",
     ["set user slack presence status text and emoji", "update slack status message away in meeting", "change profile status on slack"],
     ["set my slack status to 'In a meeting until 4pm' :calendar:", "update status to 'Lunch break'", "ändere meinen Slack Status"]),
    ("slack_archive_channel",
     ["archive completed or inactive slack channel", "close and archive old channel in workspace", "mark slack channel archived"],
     ["archive the old #project-titan channel", "close down inactive channel", "archiviere den Slack Channel"]),
    ("slack_get_replies",
     ["fetch message thread replies in slack conversation", "read all comments in slack thread ts", "get discussion thread messages"],
     ["read the thread responses under the deployment announcement", "fetch all replies to message ts 1698234.001", "hole alle Antworten im Slack Thread"]),
    ("slack_schedule_message",
     ["schedule message to be sent later in slack channel", "post delayed scheduled message to user", "queue slack message for tomorrow 9am"],
     ["schedule a reminder in #general for tomorrow 09:00", "send this greeting at 8am tomorrow", "plane eine Nachricht für morgen früh"]),
    ("slack_search_messages",
     ["search workspace chat messages with query terms in slack", "find previous conversation in slack channels", "search slack history"],
     ["search Slack for mentions of 'database migration error'", "find when someone mentioned the auth service outage", "suche im Slack Chat nach Begriffen"]),

    # --- Jira / Issue Tracker (10 tools) ---
    ("jira_create_issue",
     ["create new issue bug task story in Jira", "file ticket in jira project board", "open new jira ticket with summary"],
     ["create a high priority bug in PROJ: 'Login fails with 500'", "file a task in Jira for database cleanup", "erstelle ein neues Jira Ticket"]),
    ("jira_transition_status",
     ["move jira issue to new status in progress done closed", "transition workflow state of jira ticket", "update ticket status on board"],
     ["move ticket PROJ-123 to 'In Review'", "close issue PROJ-456 as Done", "setze den Status des Tickets auf erledigt"]),
    ("jira_add_comment",
     ["add comment or update note to jira issue", "post discussion comment on jira ticket", "leave note on ticket"],
     ["comment on PROJ-88: 'Reproduced on staging, investigating now'", "add update to ticket regarding root cause", "schreibe einen Kommentar ins Jira Ticket"]),
    ("jira_assign_issue",
     ["assign jira issue to team member assignee", "change owner of jira ticket", "reassign issue to developer"],
     ["assign PROJ-204 to user alex.meier", "assign this unassigned bug to me", "weise das Ticket dem Kollegen zu"]),
    ("jira_search_jql",
     ["search jira issues using JQL query filter", "query tickets with jira query language", "find issues by project status sprint"],
     ["JQL: project = SEC AND status = Open AND priority = Blocker", "search for all tickets updated in the last 24h", "suche Tickets per JQL"]),
    ("jira_link_issues",
     ["link two jira issues duplicates blocks relates to", "create issue link relationship in jira", "connect dependent tickets"],
     ["link PROJ-10 as blocked by PROJ-09", "mark ticket PROJ-55 as duplicate of PROJ-40", "verknüpfe die beiden Jira Tickets"]),
    ("jira_create_sprint",
     ["create a new sprint on agile jira board", "schedule upcoming sprint with start and end date", "open new iteration sprint"],
     ["create Sprint 24 for the Core Team starting next Monday", "open a new two-week sprint in Jira", "erstelle einen neuen Sprint"]),
    ("jira_log_worktime",
     ["log work time spent on jira issue tempo worklog", "record hours worked on ticket", "log 2h 30m on issue PROJ-99"],
     ["log 3.5 hours on PROJ-12 for code review", "record 45 minutes on the bug fix ticket", "buche Arbeitszeit auf das Ticket"]),
    ("jira_get_board",
     ["get agile board configuration and columns in jira", "fetch kanban or scrum board issue list", "inspect active sprint board"],
     ["show me all columns and cards on the Backend Scrum board", "fetch active sprint issues from board 14", "hole das aktuelle Kanban Board"]),
    ("jira_delete_issue",
     ["delete unwanted issue or test ticket from jira", "permanently remove ticket from project", "delete jira card"],
     ["delete the test ticket PROJ-999", "remove duplicate spam ticket from project", "lösche das Test-Ticket in Jira"]),

    # --- Docker / Containers (10 tools) ---
    ("docker_build_image",
     ["build docker container image from Dockerfile", "docker build -t image tag .", "package application into container image"],
     ["docker build -t my-app:v2.1 -f Dockerfile .", "build the container image for production", "baue das Docker Image aus dem Dockerfile"]),
    ("docker_run_container",
     ["run docker container with port mapping and env", "docker run -d -p container start", "instantiate container from image"],
     ["run redis container in background mapping port 6379", "docker run -e ENV=prod my-service", "starte einen neuen Container"]),
    ("docker_stop_container",
     ["stop running docker container gracefully", "docker stop container id or name", "halt active container process"],
     ["stop the running postgres-dev container", "docker stop c81a9f02", "halte den laufenden Container an"]),
    ("docker_inspect_container",
     ["inspect docker container json metadata ip mounts", "docker inspect container details", "view container configuration and network"],
     ["what is the IP address of container api-backend?", "inspect mounts and env vars of docker container", "zeige detaillierte Infos zum Container per docker inspect"]),
    ("docker_prune_system",
     ["prune unused docker containers images networks volumes", "docker system prune -af reclaim disk space", "clean up dangling docker cache"],
     ["clean up all unused docker images and dangling layers", "docker system prune to free up disk space", "bereinige ungenutzte Docker Images"]),
    ("docker_pull_image",
     ["pull container image from docker hub or registry", "docker pull repository:tag", "download container image from registry"],
     ["pull the latest python:3.11-slim image", "download postgres:16 from docker registry", "ziehe das Docker Image aus der Registry"]),
    ("docker_compose_up",
     ["start multi-container environment with docker compose up", "launch local services defined in compose file", "spin up docker-compose stack"],
     ["run docker compose up -d to start local test stack", "start up all services in docker-compose.yml", "starte die Compose Umgebung"]),
    ("docker_compose_down",
     ["stop and remove docker compose containers and networks", "docker compose down --volumes", "tear down local compose stack"],
     ["shut down the compose environment and remove volumes", "docker compose down", "stoppe den docker-compose Stack"]),
    ("docker_view_logs",
     ["fetch logs from docker container stdout stderr", "docker logs --tail container", "read container output stream"],
     ["show the last 50 lines of logs from container web-app", "docker logs -f my-container", "zeige die Logs des Docker Containers"]),
    ("docker_tag_image",
     ["tag local docker image for registry repository", "docker tag source_image:tag target_repo:tag", "alias container image"],
     ["tag local-build:latest as 12345.dkr.ecr.eu-west-1.amazonaws.com/app:v1", "tag the image for docker hub", "versehe das Docker Image mit einem neuen Tag"]),

    # --- FileSystem & Code (10 tools) ---
    ("fs_read_file",
     ["read contents of file on local filesystem", "open and view text file", "cat read file path"],
     ["read the contents of src/klix/engine.py lines 1-50", "open and inspect config/settings.yaml", "lies die Datei README.md"]),
    ("fs_write_file",
     ["write content to file or create new file on disk", "overwrite save code to file path", "write file buffer"],
     ["save the updated python code into ./utils.py", "create a new config file with these contents", "schreibe den Text in die Datei config.json"]),
    ("fs_ripgrep_search",
     ["search text pattern across files with ripgrep grep", "fast regex search in workspace directory", "find string in codebase"],
     ["search for 'class HybridBackbone' across all python files", "find occurrences of 'TODO:' in the codebase", "suche mit ripgrep nach allen Vorkommen von 'export'"]),
    ("fs_list_directory",
     ["list files and folders in directory path", "inspect directory tree ls dir", "browse workspace folder contents"],
     ["what files are in the ./evals folder?", "list subdirectories in src/", "zeige mir den Inhalt des Verzeichnisses"]),
    ("fs_delete_file",
     ["delete file or directory from filesystem rm", "remove file path permanently", "unlink file on disk"],
     ["delete temporary file ./scratch.txt", "remove old build artifact directory", "lösche die temporäre Datei"]),
    ("fs_move_file",
     ["move or rename file or directory on disk mv", "rename file path to new location", "relocate file in filesystem"],
     ["rename old_test.py to test_regression.py", "move the download file into ./data/", "verschiebe die Datei in den Ordner"]),
    ("fs_create_directory",
     ["create new directory folder on filesystem mkdir", "ensure path exists mkdir -p", "make new directory"],
     ["create folder ./assets/icons/", "make a new output directory for results", "erstelle einen neuen Ordner"]),
    ("fs_chmod_permissions",
     ["change file mode permissions chmod executable", "make script executable chmod +x", "modify filesystem permission bits"],
     ["chmod +x ./scripts/deploy.sh to make it runnable", "change permissions of secret.key to 600", "mache das Bash Skript ausführbar"]),
    ("fs_calculate_hash",
     ["calculate SHA256 or MD5 hash checksum of file", "verify file integrity with checksum hash", "compute file digest"],
     ["calculate sha256 checksum of the installer binary", "verify md5 hash of the downloaded zip", "berechne den SHA-256 Hash der Datei"]),
    ("fs_create_archive",
     ["create compressed tar.gz or zip archive from directory", "pack files into tarball or zip", "compress folder into archive"],
     ["compress the ./dist folder into release.tar.gz", "create a zip archive of the logs directory", "packe den Ordner in ein tar.gz Archiv"]),

    # --- Observability & Alerts (10 tools) ---
    ("prom_query_instant",
     ["execute instant Prometheus PromQL query", "query current prometheus metric value", "evaluate promql expression now"],
     ["PromQL: sum(rate(http_requests_total{status=~'5..'}[5m]))", "get current memory usage percentage per pod from Prometheus", "führe eine Prometheus PromQL Abfrage aus"]),
    ("prom_query_range",
     ["query Prometheus metrics over time range vector", "promql range query with step and start end time", "fetch historical metric time series"],
     ["plot CPU usage over the last 6 hours using prometheus range query", "fetch 24h request latency histogram series", "hole Zeitreihen-Metriken aus Prometheus"]),
    ("dd_search_logs",
     ["search logs in Datadog log management", "query datadog log events with facets and filter", "find error logs in datadog"],
     ["search Datadog for service:billing status:error in past 15m", "find logs containing 'OutOfMemoryError' in datadog", "suche in den Datadog Logs"]),
    ("dd_get_trace",
     ["retrieve APM distributed trace by trace_id in Datadog", "inspect flamegraph and span latency for trace", "fetch apm trace details"],
     ["fetch trace details for trace_id 847291048201", "view the distributed span waterfall for this failed request", "hole den Datadog APM Trace"]),
    ("dd_create_metric_alert",
     ["create Datadog monitor alert on metric threshold", "configure datadog alert rule for error rate", "set up new monitor in datadog"],
     ["alert in Datadog when error rate exceeds 2% for 5 minutes", "create a new latency monitor on endpoint /checkout", "erstelle einen Datadog Alert"]),
    ("dd_mute_alert",
     ["mute or silence Datadog monitor during maintenance", "suppress alert notifications for monitor id", "silence noisy monitor"],
     ["mute the RDS CPU monitor for 2 hours during migration", "silence alert #49102 until tomorrow morning", "schalte den Alert temporär stumm"]),
    ("pagerduty_trigger_incident",
     ["trigger high urgency PagerDuty incident to page on-call", "open pagerduty incident for critical outage", "page on-call engineer"],
     ["page the primary on-call: payment gateway is completely down", "trigger PagerDuty incident for site-wide outage", "löse einen PagerDuty Vorfall aus"]),
    ("pagerduty_acknowledge",
     ["acknowledge PagerDuty incident to halt escalation", "ack pagerduty alert indicating investigation", "acknowledge active page"],
     ["acknowledge incident #P99281 on PagerDuty", "ack the alert so it stops escalating", "bestätige den PagerDuty Incident"]),
    ("sentry_get_issue",
     ["fetch Sentry error issue stack trace and breadcrumbs", "inspect sentry crash report details", "view exception in sentry"],
     ["show stack trace and culprit for Sentry issue AUTH-E28", "fetch recent events for exception TypeError in sentry", "hole den Sentry Stacktrace"]),
    ("sentry_resolve_issue",
     ["mark Sentry issue as resolved in current release", "close error report in sentry tracking", "resolve sentry bug"],
     ["resolve issue BACKEND-401 as fixed in v2.4.0", "mark the Sentry exception as resolved", "schließe das Sentry Problem"]),

    # --- Billing, CRM & Support (10 tools) ---
    ("stripe_create_customer",
     ["create new customer profile in Stripe with email", "provision stripe customer account for billing", "add paying user to stripe"],
     ["create a Stripe customer for user 'john@corp.com'", "register new billing customer with card details", "lege einen neuen Kunden in Stripe an"]),
    ("stripe_create_invoice",
     ["generate invoice draft and finalize in Stripe", "bill customer with itemized stripe invoice", "issue invoice for enterprise subscription"],
     ["create an invoice for $5,000 for customer cus_abc123", "send invoice to customer for annual license", "erstelle eine Rechnung in Stripe"]),
    ("stripe_refund_charge",
     ["refund a credit card charge or payment in Stripe", "issue full or partial refund to customer", "process payment refund in stripe"],
     ["refund payment charge ch_3N82b partially by $50", "process full refund for transaction tx_991", "veranlasse eine Rückerstattung in Stripe"]),
    ("stripe_cancel_subscription",
     ["cancel active customer subscription in Stripe", "terminate recurring billing subscription at period end", "stop stripe subscription"],
     ["cancel subscription sub_9981 immediately without refund", "set customer subscription to cancel at period end", "kündige das Stripe Abonnement"]),
    ("stripe_list_payments",
     ["list recent payment intents and charges in Stripe", "inspect transaction history and declined payments", "query stripe charges"],
     ["show failed payment intents from the last 24 hours", "list all charges for customer cus_7712", "zeige die letzten Zahlungen in Stripe"]),
    ("hubspot_get_contact",
     ["retrieve CRM contact properties from HubSpot", "lookup lead contact by email address in crm", "fetch customer record hubspot"],
     ["lookup contact info for 'sara@enterprise.com' in HubSpot", "what is the deal stage of lead Markus in CRM?", "hole die Kontaktdaten aus HubSpot"]),
    ("hubspot_update_deal",
     ["update sales deal stage amount or owner in HubSpot CRM", "move hubspot deal to closed won", "modify crm deal record"],
     ["move deal 'Acme Enterprise 100 Seats' to Closed-Won", "update deal amount to $120,000 in HubSpot", "aktualisiere den Deal im CRM"]),
    ("zendesk_create_ticket",
     ["create customer support ticket in Zendesk", "open new zendesk support case with subject and body", "submit support inquiry"],
     ["create high priority ticket in Zendesk: 'Customer cannot log in'", "file support request for account unlock", "erstelle ein Ticket im Zendesk"]),
    ("zendesk_update_ticket",
     ["update zendesk ticket status priority or add public reply", "change support ticket state to solved", "reply to zendesk ticket"],
     ["mark Zendesk ticket #10492 as solved with comment 'Resolved via reset'", "update ticket priority to urgent", "aktualisiere das Support Ticket"]),
    ("sendgrid_send_email",
     ["send transactional email notification via SendGrid API", "dispatch email template to recipient", "send customer notification mail"],
     ["send password reset email to user@test.com via SendGrid", "send email receipt with invoice PDF attached", "sende eine E-Mail über SendGrid"]),
]

def run_benchmark():
    print("=" * 70)
    print(f"BENCHMARK: 100 MCP TOOLS DYNAMIC RETRIEVAL WITH KLIX")
    print(f"Total Tools: {len(TOOLS_DATA)}")
    print("=" * 70)

    # 1. Build anchors dictionary: {tool_name: [anchors]}
    anchors = {tool_name: tool_anchors for tool_name, tool_anchors, _ in TOOLS_DATA}
    total_anchors = sum(len(a) for a in anchors.values())
    print(f"Total reference anchors registered: {total_anchors} (avg {total_anchors/len(anchors):.1f} per tool)")

    # 2. Collect test queries: [(test_query, ground_truth_tool_name)]
    test_cases = []
    for tool_name, _, queries in TOOLS_DATA:
        for q in queries:
            test_cases.append((q, tool_name))
    print(f"Total test queries (unseen, multilingual EN/DE): {len(test_cases)}")

    # 3. Test two classifiers: Centroid vs Nearest
    for classifier in ["centroid", "nearest"]:
        print(f"\n--- Testing Classifier: {classifier.upper()} ---")
        engine = DecisionEngine(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        engine.add_head(
            Choice(
                name="tool",
                options=anchors,
                classifier=classifier,
                keyword_boost=0.5,
            )
        )

        t0_compile = time.perf_counter()
        engine.compile()
        t_compile = (time.perf_counter() - t0_compile) * 1000
        print(f"Compile time: {t_compile:.1f} ms")

        # Evaluate all queries
        latencies = []
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        top10_hits = 0

        for query, expected_tool in test_cases:
            t0 = time.perf_counter()
            res = engine.decide(query)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            # Extract full ranked list from details("tool")["scores"]
            scores_dict = res.details("tool").get("scores", {})
            # Rank tools by score descending
            ranked_tools = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)
            ranked_names = [name for name, score in ranked_tools]

            if ranked_names and ranked_names[0] == expected_tool:
                top1_hits += 1
            if expected_tool in ranked_names[:3]:
                top3_hits += 1
            if expected_tool in ranked_names[:5]:
                top5_hits += 1
            if expected_tool in ranked_names[:10]:
                top10_hits += 1

        n = len(test_cases)
        lat_arr = np.array(latencies)
        print(f"Recall@1 (Top-1 Accuracy):  {top1_hits/n*100:5.1f}%  ({top1_hits}/{n})")
        print(f"Recall@3 (Top-3 Retrieval): {top3_hits/n*100:5.1f}%  ({top3_hits}/{n})")
        print(f"Recall@5 (Top-5 Retrieval): {top5_hits/n*100:5.1f}%  ({top5_hits}/{n})")
        print(f"Recall@10 (Top-10 Retrieval): {top10_hits/n*100:5.1f}% ({top10_hits}/{n})")
        print(f"Latency per query (CPU):")
        print(f"  Mean:   {lat_arr.mean():.2f} ms")
        print(f"  Median: {np.median(lat_arr):.2f} ms")
        print(f"  P95:    {np.percentile(lat_arr, 95):.2f} ms")
        print(f"  Min:    {lat_arr.min():.2f} ms")

    print("\n" + "=" * 70)
    print("DEMO: Inspecting Top-5 Tools for 4 sample queries:")
    sample_queries = [
        "SELECT email FROM users WHERE id = 42",
        "Der Pod im Cluster stürzt dauernd mit CrashLoopBackOff ab, warum?",
        "Schick eine kurze Nachricht in den Engineering Channel",
        "Revert the last bad commit and push branch to origin",
    ]
    for sq in sample_queries:
        res = engine.decide(sq)
        ranked = sorted(res.details("tool")["scores"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"\nQuery: '{sq}' (Latency: {res.latency_ms:.1f}ms)")
        for rank, (tname, sc) in enumerate(ranked, 1):
            print(f"  #{rank} [{sc:.3f}] {tname}")

if __name__ == "__main__":
    run_benchmark()
