pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "robsonultramari/meu-projeto"
        // O nome da branch entra na tag da imagem (ex.: homolog-5)
        DOCKER_TAG   = "${env.BRANCH_NAME}-${env.BUILD_NUMBER}"
    }

    stages {
        stage('Definir ambiente') {
            steps {
                script {
                    // Mapa: branch -> nome do ambiente e porta publicada
                    def ambientes = [
                        'projeto01': [nome: 'dev',      porta: '3001'],
                        'homolog'  : [nome: 'homolog',  porta: '3002'],
                        'preprod'  : [nome: 'preprod',  porta: '3003'],
                        'main'     : [nome: 'producao', porta: '3000']
                    ]
                    def amb = ambientes[env.BRANCH_NAME]
                    if (amb == null) {
                        error "Branch ${env.BRANCH_NAME} sem ambiente configurado"
                    }
                    env.AMBIENTE = amb.nome
                    env.PORTA    = amb.porta
                    echo "Branch ${env.BRANCH_NAME} -> ambiente ${env.AMBIENTE} (porta ${env.PORTA})"
                }
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

        stage('Deploy no container do ambiente') {
            steps {
                // Remove o container antigo do ambiente e sobe o novo com a imagem recém-criada
                sh """
                    docker rm -f app-${AMBIENTE} || true
                    docker run -d --name app-${AMBIENTE} \
                        --log-driver json-file -p ${PORTA}:3000 \
                        -e AMBIENTE=${AMBIENTE} \
                        ${DOCKER_IMAGE}:${DOCKER_TAG}
                """
            }
        }
    }

    post {
        success { echo "Deploy concluído em ${env.AMBIENTE}" }
        failure { echo 'Falha no pipeline.' }
    }
}
