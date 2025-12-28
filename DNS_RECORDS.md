# DNS Records for goeckoh.com

This document contains all DNS records needed to configure the goeckoh.com domain with GoDaddy or any DNS provider.

## Table of Contents
- [A Records](#a-records)
- [CNAME Records](#cname-records)
- [MX Records (Email)](#mx-records-email)
- [TXT Records](#txt-records)
- [Nameservers](#nameservers)
- [DNSSEC DS Records](#dnssec-ds-records)
- [Domain Forwarding](#domain-forwarding)
- [Subdomain Configuration](#subdomain-configuration)

---

## A Records

A records map domain names to IPv4 addresses. Replace `YOUR_SERVER_IP` with your actual server IP address.

| Type | Host | Value | TTL |
|------|------|-------|-----|
| A | @ | YOUR_SERVER_IP (e.g., 185.199.108.153) | 600 |
| A | www | YOUR_SERVER_IP (e.g., 185.199.108.153) | 600 |
| A | api | YOUR_API_SERVER_IP | 600 |
| A | docs | YOUR_DOCS_SERVER_IP | 600 |
| A | blog | YOUR_BLOG_SERVER_IP | 600 |

**Note for GitHub Pages:**
If hosting on GitHub Pages, use these A records instead:
- 185.199.108.153
- 185.199.109.153
- 185.199.110.153
- 185.199.111.153

---

## CNAME Records

CNAME records create aliases for your domain. The host "@" cannot be a CNAME.

| Type | Host | Value | TTL |
|------|------|-------|-----|
| CNAME | www | goeckoh.com | 3600 |
| CNAME | api | goeckoh-api.herokuapp.com (or your API host) | 3600 |
| CNAME | docs | kaleidoscopeai.github.io | 3600 |
| CNAME | blog | goeckoh-blog.netlify.app (or your blog host) | 3600 |
| CNAME | download | goeckoh.com | 3600 |
| CNAME | support | goeckoh.com | 3600 |
| CNAME | app | goeckoh-app.vercel.app (or your app host) | 3600 |
| CNAME | cdn | goeckoh.b-cdn.net (or your CDN) | 3600 |
| CNAME | mail | ghs.googlehosted.com (if using Google Workspace) | 3600 |

---

## MX Records (Email)

MX records route email to your email servers. Configure based on your email provider.

### For Gmail/Google Workspace:

| Type | Host | Value | Priority | TTL |
|------|------|-------|----------|-----|
| MX | @ | aspmx.l.google.com | 1 | 3600 |
| MX | @ | alt1.aspmx.l.google.com | 5 | 3600 |
| MX | @ | alt2.aspmx.l.google.com | 5 | 3600 |
| MX | @ | alt3.aspmx.l.google.com | 10 | 3600 |
| MX | @ | alt4.aspmx.l.google.com | 10 | 3600 |

### For Microsoft 365:

| Type | Host | Value | Priority | TTL |
|------|------|-------|----------|-----|
| MX | @ | goeckoh-com.mail.protection.outlook.com | 0 | 3600 |

---

## TXT Records

TXT records are used for domain verification, SPF, DKIM, and DMARC.

| Type | Host | Value | TTL |
|------|------|-------|-----|
| TXT | @ | v=spf1 include:_spf.google.com ~all | 3600 |
| TXT | @ | google-site-verification=YOUR_VERIFICATION_CODE | 3600 |
| TXT | _dmarc | v=DMARC1; p=quarantine; rua=mailto:dmarc@goeckoh.com | 3600 |
| TXT | @ | github-domain-verification=YOUR_GITHUB_VERIFICATION | 3600 |

**SPF Record Explanation:**
- `v=spf1` - SPF version 1
- `include:_spf.google.com` - Allow Google servers to send email
- `~all` - Soft fail for all other servers

**DMARC Record Explanation:**
- `v=DMARC1` - DMARC version 1
- `p=quarantine` - Quarantine emails that fail authentication
- `rua=mailto:dmarc@goeckoh.com` - Send aggregate reports to this email

---

## Nameservers

Nameservers control DNS for your domain. You have two options:

### Option 1: Use GoDaddy's Nameservers (Default)
```
ns01.domaincontrol.com
ns02.domaincontrol.com
```

### Option 2: Use Cloudflare Nameservers (Recommended for better performance)
```
aldo.ns.cloudflare.com
lola.ns.cloudflare.com
```

### Option 3: Use Custom Nameservers
If you're using a custom DNS provider, replace with your nameserver addresses.

---

## DNSSEC DS Records

DNSSEC adds security to DNS by digitally signing records. Generate these from your DNS provider or domain registrar.

**Example DS Record Format:**
```
Key Tag: 12345
Algorithm: 13 (ECDSAP256SHA256)
Digest Type: 2 (SHA-256)
Digest: 1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF
```

**To generate DNSSEC records:**
1. Enable DNSSEC in your DNS provider (GoDaddy, Cloudflare, etc.)
2. Copy the DS record details provided
3. Add the DS record to your domain registrar

**GoDaddy DNSSEC Setup:**
1. Log into GoDaddy account
2. Go to My Products > DNS
3. Click on "Manage DNSSEC"
4. Enable DNSSEC and copy the DS record

---

## Domain Forwarding

Configure domain forwarding to redirect traffic between domains/subdomains.

### HTTP/HTTPS Forwarding Rules:

| From | To | Type | Status Code |
|------|-----|------|-------------|
| goeckoh.com | https://www.goeckoh.com | Permanent | 301 |
| http://www.goeckoh.com | https://www.goeckoh.com | Permanent | 301 |
| http://goeckoh.com | https://www.goeckoh.com | Permanent | 301 |

### Subdomain Forwarding:

| From | To | Type |
|------|-----|------|
| support.goeckoh.com | https://www.goeckoh.com/support | Permanent (301) |
| privacy.goeckoh.com | https://www.goeckoh.com/privacy | Permanent (301) |
| terms.goeckoh.com | https://www.goeckoh.com/terms | Permanent (301) |

---

## Subdomain Configuration

List of recommended subdomains and their purposes:

| Subdomain | Purpose | Example Target |
|-----------|---------|----------------|
| www | Main website | goeckoh.com |
| api | API endpoints | api.goeckoh.com |
| app | Web application | app.goeckoh.com |
| docs | Documentation | docs.goeckoh.com |
| blog | Blog/Articles | blog.goeckoh.com |
| download | Download page | download.goeckoh.com |
| support | Support portal | support.goeckoh.com |
| cdn | Content delivery | cdn.goeckoh.com |
| mail | Email webmail | mail.goeckoh.com |
| dev | Development environment | dev.goeckoh.com |
| staging | Staging environment | staging.goeckoh.com |
| test | Testing environment | test.goeckoh.com |
| demo | Demo application | demo.goeckoh.com |
| assets | Static assets | assets.goeckoh.com |
| images | Image hosting | images.goeckoh.com |
| media | Media files | media.goeckoh.com |
| status | Status page | status.goeckoh.com |
| analytics | Analytics dashboard | analytics.goeckoh.com |

---

## Setup Instructions for GoDaddy

### Step 1: Access DNS Management
1. Log into your GoDaddy account
2. Navigate to **My Products**
3. Find **goeckoh.com** and click **DNS**

### Step 2: Add A Records
1. Click **Add** button
2. Select **A** from Type dropdown
3. Enter Host (@ for root, www for www subdomain)
4. Enter Points to (your IP address)
5. Set TTL (600 seconds recommended)
6. Click **Save**

### Step 3: Add CNAME Records
1. Click **Add** button
2. Select **CNAME** from Type dropdown
3. Enter Host (subdomain name)
4. Enter Points to (target domain)
5. Set TTL (3600 seconds recommended)
6. Click **Save**

### Step 4: Add MX Records
1. Click **Add** button
2. Select **MX** from Type dropdown
3. Enter Host (@)
4. Enter Points to (mail server)
5. Set Priority (1 for primary, 5-10 for backups)
6. Click **Save**

### Step 5: Add TXT Records
1. Click **Add** button
2. Select **TXT** from Type dropdown
3. Enter Host (@ or subdomain)
4. Enter TXT Value (verification or SPF string)
5. Click **Save**

### Step 6: Configure Nameservers (Optional)
1. In DNS Management, scroll to **Nameservers** section
2. Click **Change**
3. Select **Custom** if using non-GoDaddy nameservers
4. Enter nameserver addresses
5. Click **Save**

### Step 7: Enable DNSSEC (Optional but Recommended)
1. In DNS Management, find **DNSSEC** section
2. Click **Manage DNSSEC**
3. Enable DNSSEC
4. Copy the DS record details
5. These are automatically added by GoDaddy

### Step 8: Set Up Forwarding
1. In My Products, click on **goeckoh.com**
2. Click **Settings**
3. Find **Forwarding** section
4. Click **Add Forwarding**
5. Enter From domain and To URL
6. Select Forward Type (301 Permanent or 302 Temporary)
7. Click **Save**

---

## DNS Propagation

After making DNS changes, allow time for propagation:
- **Local ISP**: 2-4 hours
- **Global**: 24-48 hours
- **Complete**: Up to 72 hours

Check DNS propagation status at:
- https://www.whatsmydns.net/
- https://dnschecker.org/
- https://www.dnswatch.info/

---

## Verification

After setup, verify your DNS records:

```bash
# Check A record
dig goeckoh.com A
dig www.goeckoh.com A

# Check MX records
dig goeckoh.com MX

# Check TXT records
dig goeckoh.com TXT

# Check CNAME
dig api.goeckoh.com CNAME

# Check nameservers
dig goeckoh.com NS

# Check DNSSEC
dig goeckoh.com +dnssec
```

---

## Troubleshooting

### Common Issues:

1. **DNS not resolving**
   - Wait 24-48 hours for propagation
   - Clear DNS cache: `ipconfig /flushdns` (Windows) or `sudo dscacheutil -flushcache` (Mac)
   - Check if nameservers are correct

2. **Email not working**
   - Verify MX records are correct
   - Check SPF record syntax
   - Ensure priority values are set correctly

3. **HTTPS not working**
   - Verify A records point to correct IP
   - Check SSL certificate is installed on server
   - Enable HTTPS redirect in server configuration

4. **Subdomain not resolving**
   - Verify CNAME or A record exists
   - Check for typos in host name
   - Wait for DNS propagation

---

## Security Recommendations

1. **Enable DNSSEC** - Protects against DNS spoofing
2. **Use HTTPS** - Encrypt traffic with SSL/TLS certificate
3. **Implement SPF** - Prevent email spoofing
4. **Add DMARC** - Email authentication and reporting
5. **Set up DKIM** - Email signature verification
6. **Regular Audits** - Review DNS records quarterly
7. **Monitor Changes** - Set up alerts for DNS modifications

---

## Contact Information

For DNS support:
- **Technical Contact**: support@goeckoh.com
- **Administrative Contact**: admin@goeckoh.com
- **Abuse Contact**: abuse@goeckoh.com

---

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-12-28 | 1.0 | Initial DNS configuration | Goeckoh Team |

---

## Additional Resources

- [GoDaddy DNS Management Guide](https://www.godaddy.com/help/manage-dns-680)
- [Cloudflare DNS Setup](https://developers.cloudflare.com/dns/)
- [GitHub Pages Custom Domain](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site)
- [Google Workspace MX Records](https://support.google.com/a/answer/174125)
- [SPF Record Syntax](https://www.dmarcanalyzer.com/spf/spf-record-syntax/)
- [DMARC Policy Generator](https://www.kitterman.com/dmarc/assistant.html)

---

**Last Updated**: December 28, 2025
**Maintained By**: Goeckoh Technical Team
