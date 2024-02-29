# update_repo.ps1
# powershell -ExecutionPolicy ByPass -File .\update_repo.ps1

# Pregunta al usuario por el mensaje de commit
$commitMessage = Read-Host -Prompt "Ingresa tu mensaje de commit"

# # Navega al directorio de tu proyecto (ajusta la ruta según sea necesario)
# Set-Location "C:\ruta\a\tu\proyecto"

# Agrega todos los archivos cambiados al staging area
git add .

# Hace el commit con el mensaje proporcionado
git commit -m $commitMessage

# Empuja los cambios al repositorio remoto
git push origin main

Write-Host "Cambios subidos exitosamente."


# Guarda el contenido en un archivo llamado update_repo.ps1.
# Abre PowerShell y navega al directorio donde guardaste el script.
# Ejecuta el script con .\update_repo.ps1.