pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "robsonultramari/meu-projeto"
        DOCKER_TAG   = "${env.BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
                    sh "docker push ${DOCKER_IMAGE}:${DOCKER_TAG}"
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh "kubectl set image deployment/meu-app meu-app=${DOCKER_IMAGE}:${DOCKER_TAG}"
            }
        }
    }

    post {
        success { echo '✅ Pipeline concluído com sucesso!' }
        failure { echo '❌ Falha no pipeline.' }
    }
}
