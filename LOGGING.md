# Logging Documentation

## Overview

The application uses a comprehensive logging system that stores all logs in the `./logs/` directory. This includes errors, warnings, API requests, and general application events.

## Log Files

The application creates three main log files:

### 1. `logs/app.log`
**Purpose**: All application logs (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Contains**:
- Service initialization events
- Document processing logs
- Voice service operations
- RAG queries and responses
- General application flow

**Example entries**:
```
2025-01-13 14:30:15 - apps.api.main - INFO - [main.py:130] - 🚀 All services initialized successfully!
2025-01-13 14:31:22 - apps.api.services.document.processor - DEBUG - [processor.py:45] - Processing document abc123
```

### 2. `logs/error.log`
**Purpose**: ERROR and CRITICAL level logs only

**Contains**:
- Application errors
- Service failures
- Database connection issues
- Processing errors
- Exception tracebacks

**Example entries**:
```
2025-01-13 14:32:10 - apps.api.services.rag.llamaindex_service - ERROR - [llamaindex_service.py:123] - Query processing failed: Connection timeout
Traceback (most recent call last):
  ...
```

### 3. `logs/access.log`
**Purpose**: HTTP request/response logs

**Contains**:
- All API requests
- Response status codes
- Request duration
- Client IP addresses

**Example entries**:
```
2025-01-13 14:35:42 - access - INFO - [main.py:230] - [a3f5e1c2] POST /api/v1/documents/upload - Client: 127.0.0.1
2025-01-13 14:35:45 - access - INFO - [main.py:243] - [a3f5e1c2] POST /api/v1/documents/upload - Status: 200 - Duration: 3.124s
```

## Log Rotation

All log files use **rotating file handlers**:
- **Max size per file**: 10 MB
- **Backup count**: 5 files
- **Total storage**: Up to 50 MB per log type

When a log file reaches 10 MB, it's automatically renamed (e.g., `app.log.1`) and a new `app.log` is created.

## Viewing Logs

### Quick Commands

**View last 50 lines of app log:**
```bash
python scripts/view_logs.py tail
```

**View all errors:**
```bash
python scripts/view_logs.py --file error tail
```

**Filter for specific severity:**
```bash
python scripts/view_logs.py filter --severity ERROR
python scripts/view_logs.py filter --severity WARNING
```

**View recent logs (last hour):**
```bash
python scripts/view_logs.py recent --hours 1
```

**View recent logs (last 24 hours):**
```bash
python scripts/view_logs.py recent --hours 24
```

**Search for specific pattern:**
```bash
python scripts/view_logs.py search "document.*failed"
python scripts/view_logs.py search "ImportError" --case-sensitive
```

**Error summary:**
```bash
python scripts/view_logs.py --file error summary
```

### Using Standard Unix Tools

**Tail logs in real-time:**
```bash
tail -f logs/app.log
tail -f logs/error.log
```

**Count errors:**
```bash
grep -c "ERROR" logs/app.log
```

**View specific time range:**
```bash
grep "2025-01-13 14:" logs/app.log
```

**Find all errors with context:**
```bash
grep -B 2 -A 2 "ERROR" logs/app.log
```

## Log Format

### Console Output (Simple Format)
```
2025-01-13 14:30:15 - INFO - Service started
```

### File Output (Detailed Format)
```
2025-01-13 14:30:15 - apps.api.main - INFO - [main.py:130] - Service started
```

**Format breakdown**:
- `2025-01-13 14:30:15` - Timestamp
- `apps.api.main` - Logger name (module path)
- `INFO` - Log level
- `[main.py:130]` - Source file and line number
- `Service started` - Log message

## Configuration

Logging is configured in [apps/api/main.py](apps/api/main.py):

**Log levels** (from most to least verbose):
- `DEBUG` - Detailed information for debugging
- `INFO` - General informational messages
- `WARNING` - Warning messages (something unexpected)
- `ERROR` - Error messages (operation failed)
- `CRITICAL` - Critical errors (application may crash)

**To change log level**, modify the environment variable:
```bash
export LOG_LEVEL=DEBUG  # More verbose
export LOG_LEVEL=INFO   # Default
export LOG_LEVEL=WARNING # Less verbose
```

Or in `.env` file:
```
LOG_LEVEL=DEBUG
```

## Request Tracking

All HTTP requests are assigned a unique request ID for tracking:

```
[a3f5e1c2] POST /api/v1/query - Client: 192.168.1.100
[a3f5e1c2] POST /api/v1/query - Status: 200 - Duration: 0.523s
```

The same request ID (`a3f5e1c2`) appears in all logs related to that request, making it easy to trace the entire request lifecycle.

## Error Tracking

### Finding Recent Errors

**Last 100 errors:**
```bash
tail -100 logs/error.log
```

**Errors in the last hour:**
```bash
python scripts/view_logs.py --file error recent --hours 1
```

### Error Analysis

**Group errors by type:**
```bash
python scripts/view_logs.py --file error summary
```

**Find specific error:**
```bash
grep "ModuleNotFoundError" logs/error.log
```

**Count occurrences:**
```bash
grep -c "ConnectionError" logs/error.log
```

## Best Practices

### 1. Monitor Error Log Regularly
```bash
# Set up a cron job or check daily
tail -50 logs/error.log
```

### 2. Search for Patterns
When debugging, search for specific errors:
```bash
python scripts/view_logs.py search "document_id.*abc123"
```

### 3. Check Request Duration
Find slow requests:
```bash
grep "Duration:" logs/access.log | grep -E "Duration: [0-9]{2,}\."
```

### 4. Archive Old Logs
```bash
# Backup logs before they rotate
mkdir -p logs/archive/$(date +%Y-%m)
cp logs/*.log logs/archive/$(date +%Y-%m)/
```

## Production Recommendations

For production environments, consider:

1. **External logging service**: Send logs to services like:
   - ELK Stack (Elasticsearch, Logstash, Kibana)
   - Splunk
   - DataDog
   - CloudWatch (AWS)
   - Google Cloud Logging

2. **Log aggregation**: Centralize logs from multiple instances

3. **Alerting**: Set up alerts for critical errors
   ```python
   # Example: Send email on critical errors
   import logging
   from logging.handlers import SMTPHandler
   ```

4. **Metrics**: Track error rates and response times

## Troubleshooting

### Logs not appearing?

Check if logs directory exists:
```bash
ls -la logs/
```

Create if missing:
```bash
mkdir -p logs
```

### Permission issues?

Ensure write permissions:
```bash
chmod 755 logs
chmod 644 logs/*.log
```

### Disk space full?

Check log sizes:
```bash
du -h logs/
```

Clean old logs:
```bash
rm logs/*.log.[2-9]
```

## Examples

### Debug Document Processing Issue

```bash
# Find the document ID
python scripts/view_logs.py search "document_id.*abc123"

# Check for errors related to that document
python scripts/view_logs.py search "abc123" --file error

# View timeline of events
python scripts/view_logs.py search "abc123" | grep -E "(Processing|Completed|Failed)"
```

### Monitor API Performance

```bash
# Watch requests in real-time
tail -f logs/access.log

# Find slow requests (>5 seconds)
grep "Duration:" logs/access.log | awk '$NF > 5.0'

# Count requests by endpoint
grep "Status:" logs/access.log | awk '{print $4}' | sort | uniq -c
```

### Investigate Startup Issues

```bash
# Check initialization
python scripts/view_logs.py search "initialized"

# Look for connection failures
python scripts/view_logs.py search "connection.*failed" -i
```

## Support

For issues with logging:
1. Check this documentation
2. Review [apps/api/main.py](apps/api/main.py) logging configuration
3. Verify file permissions on `logs/` directory
4. Ensure disk space is available
