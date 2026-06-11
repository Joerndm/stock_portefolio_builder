#!/bin/bash
set -euo pipefail

mysql_cmd=(mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" "${MYSQL_DATABASE}")

if [[ -f /seed-repo/database_backup.sql ]]; then
  echo "[mysql-init] Importing database_backup.sql into ${MYSQL_DATABASE}"
  "${mysql_cmd[@]}" < /seed-repo/database_backup.sql
else
  echo "[mysql-init] database_backup.sql not found. Importing database_files/ddl.sql instead."
  "${mysql_cmd[@]}" < /seed-repo/database_files/ddl.sql
fi
