import paramiko
import subprocess
import time

from openbox import logger

class TuneSSH:
    def __init__(self, server_ip="192.168.90.19", server_port=22, server_user='root', server_passwd='root'):

        self.server_ip = server_ip
        self.server_port = server_port

        self.cli_server = paramiko.SSHClient()
        self.cli_server.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.cli_server.connect(hostname=self.server_ip, port=self.server_port, username=server_user, password=server_passwd)

        self.ftp_server = self.cli_server.open_sftp()  # 定义一个ftp实例

        # transport = paramiko.Transport((self.server_ip, self.server_port))
        # transport.connect(username=server_user, password=server_passwd)
        # self.ftp_server = paramiko.SFTPClient.from_transport(transport)  # 定义一个ftp实例


    def get_file(self, remote_path: str = None, local_path: str = None):
        try:
            self.ftp_server.get(remotepath=remote_path, localpath=local_path)
            return True
        except:
            return False

    def put_file(self, local_path: str = None, remote_path: str = None):
        self.ftp_server.put(localpath=local_path, remotepath=remote_path)
        try:
            self.ftp_server.put(localpath=local_path, remotepath=remote_path)
            return True
        except:
            return False

    def exec_command(self, command: str, id='server'):
        assert id in ['server', 'local', 'local_collect']
        if id == 'server':
            ssh = self.cli_server

            stdin, stdout, stderr = ssh.exec_command(command)
            results = stdout.read().decode('utf-8')
            return_code = stdout.channel.recv_exit_status()
            error = stderr.read().decode('utf-8')
            if return_code != 0:
                print(f"Error output: {error}")

            # logger.warn("return_code: %d, result: %s" % (return_code, results))
            return results, return_code
        elif id == 'local':
            # result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # return_code = result.returncode
            start_time = time.time()
            p = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            totbuf = ''
            while True:
                try:
                    buf = p.stdout.readline()
                    if buf != '':
                        totbuf += buf
                        logger.info(buf[:-1])
                except:
                    pass
                if buf == '' and p.poll() is not None or time.time() - start_time >= 3600:
                    break
                
            # return result.stdout, return_code
            if time.time() - start_time < 3600:
                return totbuf, 0
            else:
                return totbuf, -1
        elif id == 'local_collect':
            
            p = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return p
            
            

    def close_ssh(self):
        self.ftp_server.close()
        self.cli_server.close()
