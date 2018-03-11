import os
import yaml

CONF_ENV_VAR = 'CONF_SANDBOX'

try:
    with open(os.environ[CONF_ENV_VAR]) as f:
        conf_sandbox = yaml.load(f)
except KeyError as e:
    raise ImportError("Error, %s env var not set" % CONF_ENV_VAR)
except FileNotFoundError as e:
    raise ImportError("Error opening '%s'" % os.environ[CONF_ENV_VAR])
except yaml.scanner.ScannerError as e:
    raise ImportError("Error, invalid yml: %r" % e)
