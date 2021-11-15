import sys
import shutil
import subprocess
from pathlib import Path

import click
from ..jobsdb import JobsDB
from ..func import ExPyRe
from .. import config


@click.group("expyre")
@click.option("--dbfile", help="Database file from here, rather than default")
@click.pass_context
def cli(ctx, dbfile):
    if dbfile is not None:
        config.db = JobsDB(dbfile)


def _get_jobs(**kwargs):
    kwargs = kwargs.copy()
    for k, v in list(kwargs.items()):
        if v is not None:
            kwargs[k] = [item.strip() for item in v.split(',')]

    return list(config.db.jobs(**kwargs, readable=True))


@cli.command("ls")
@click.option("--id", "-i", help="comma separated list of regexps for entire job id")
@click.option("--name", "-n", help="comma separated list of regexps for entire job name")
@click.option("--status", "-s", help="comma separated list of status values to include")
@click.option("--system", "-S", help="comma separated list of regexps for entire system name")
@click.pass_context
def cli_ls(ctx, id, name, status, system):
    """List jobs fitting criteria (default all jobs)
    """
    jobs = _get_jobs(id=id, name=name, status=status, system=system)

    if len(jobs) == 0:
        print(f"No matching jobs in JobsDB at {config.db.db_filename}")
    else:
        print("Jobs:")
        for job in jobs:
            print(job)


@cli.command("rm")
@click.option("--id", "-i", help="comma separated list of regexps for entire job id")
@click.option("--name", "-n", help="comma separated list of regexps for entire job name")
@click.option("--status", "-s", help="comma separated list of status values to include")
@click.option("--system", "-S", help="comma separated list of regexps for entire system name")
@click.option("--yes", "-y", is_flag=True, help="assume 'yes' for all confirmations, i.e. delete job and stage dirs without asking user")
@click.option("--clean", "-c", is_flag=True, help="delete local and remote stage directories")
@click.pass_context
def cli_rm(ctx, id, name, status, system, yes, clean):
    """Delete jobs fitting criteria (at least one criterion required)
    """
    if id is None and name is None and status is None and system is None:
        sys.stderr.write('At least one selection criterion is required\n\n')
        sys.stderr.write(ctx.get_help()+'\n')
        sys.exit(1)

    jobs = _get_jobs(id=id, name=name, status=status, system=system)

    for xpr in ExPyRe.from_jobsdb(jobs):
        if not yes:
            # ask user
            answer = None
            while answer not in ['y', 'n']:
                answer = input(f'Deleting "{xpr}"\nEnter n to reject or y to accept: ')
            if answer != 'y':
                sys.stderr.write(f'Not deleting "{xpr.id}"\n')
                sys.stderr.write('\n')
                continue

        xpr.clean(wipe=True, dry_run=not clean)
        config.db.remove(xpr.id)
        if not clean:
            sys.stderr.write('\n')


@cli.command("db_unlock")
@click.pass_context
def cli_db_unlock(ctx):
    """Unlock the database if it is stuck in a locked state

    This may happen if a process crashes during database access.
    Note that if another process is active and trying to access database,
    its behavior may be inconsistent.
    """

    config.db.unlock()
