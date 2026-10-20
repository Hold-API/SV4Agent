# Bundled API documentation assets

The server serves these files directly. No npm install, CDN request, or asset
download is needed when starting the server or opening `/docs` and `/redoc`.

- Swagger UI: `swagger-ui-dist@5.30.2`, Apache-2.0.
  Source: https://www.npmjs.com/package/swagger-ui-dist/v/5.30.2
  License and notice: `swagger-ui.LICENSE`, `swagger-ui.NOTICE`.
- ReDoc: `redoc@2.5.2`, MIT.
  Source: https://www.npmjs.com/package/redoc/v/2.5.2
  License: `redoc.LICENSE`.

`manifest.json` records the npm tarball URLs, SHA-512 package integrity values,
and SHA-256 checksums of the vendored files. Package integrity was verified
before extracting only these named assets from the official npm archives.

When updating, select an explicit version, verify its archive integrity, copy
the listed assets and license files, and regenerate the manifest checksums.
Keep all runtime assets in this directory when deploying the Python project.
