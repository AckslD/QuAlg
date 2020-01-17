echo "Setting up deployment with travis CI"

if !(hash travis 2> /dev/null); then
    echo "Error: Please first install travis by 'gem install travis' or 'brew install travis' (macos)"
fi

# PyPi token
echo "Create a PyPi API token if not yet done so, see https://pypi.org/help/#apitoken"
read -p "Please enter the token here..." PYPI_TOKEN

echo "Adding encrypted PyPi token..."
travis encrypt $PYPI_TOKEN --add deploy.password

# Github token
echo "Create a personal access token for GitHub if not yet done so, see https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line"
read -p "Please enter the token here..." GITHUB_TOKEN

echo "Adding encrypted GitHub token..."
travis encrypt GITHUB_TOKEN=$GITHUB_TOKEN --add deploy.password
