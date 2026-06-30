# Security Hardening

## Security Requirements

When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than `640` to prevent privilege escalation and similar security issues.

Software code, model weights, and other files downloaded from external sources may pose security risks. You must ensure that the related functions are secure.

## Hardening Precautions

The security hardening measures listed in this document are basic recommendations. You should re-evaluate the network security hardening measures for the entire system based on your own business needs. When necessary, you can refer to industry best practices and the advice of security experts.

## OS Security Hardening

### Firewall Configuration

After the operating system is installed, if common users are configured, you can add `ALWAYS_SET_PATH=yes` to the `/etc/login.defs` file to prevent unauthorized operations.

### Setting umask

You are advised to set the umask on the host and in containers to `027` or a more restrictive value to tighten file permissions.

To set umask to `027`:

1. Log in to the server as the root user and edit the `/etc/profile` file.

    ```bash
    vim /etc/profile
    ```

2. Add `umask 027` to the end of the `/etc/profile` file, then save and exit.
3. Run the following command to apply the configuration.

    ```bash
    source /etc/profile
    ```

### Ownerless File Hardening

Because official Docker images differ from the operating system on the physical machine, system users may not correspond one to one. Therefore, files created on the physical machine or in containers can become ownerless.

You can run `find / -nouser -o -nogroup` to locate ownerless files in a container or on the physical machine. Create corresponding users and groups based on the file UID and GID, or adjust existing user UIDs and group GIDs to match them. Then assign ownership to the files and prevent ownerless files from creating security risks for the system.

### Port Scanning

Monitor ports that listen on all interfaces and any unnecessary ports. If you find unnecessary ports, close them promptly. You are advised to disable insecure services, such as Telnet and FTP. For details about how to disable them, see the relevant documentation for the operating system in use.

### Anti-DoS Protection

You can protect the system against DoS attacks by rate-limiting connections from each IP address to the server. Methods include, but are not limited to, using the Linux `iptables` firewall for prevention and optimizing `sysctl` parameters. For details about how to use these methods, refer to related materials.
